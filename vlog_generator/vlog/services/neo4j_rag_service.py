"""Neo4j 地點讀取與 E5 語意候選檢索。此模組不建立索引、不寫入資料。"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
from typing import Any

# Windows：先初始化 Torch，再延遲匯入 Sentence Transformers。
import torch
from neo4j import GraphDatabase, READ_ACCESS, unit_of_work

logger = logging.getLogger(__name__)
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "playtaiwandb")
NEO4J_QUERY_TIMEOUT = float(os.getenv("NEO4J_QUERY_TIMEOUT", "30"))
E5_MODEL_NAME = "intfloat/multilingual-e5-large"
E5_DIMENSION = 1024
_driver = None
_driver_lock = threading.Lock()
_embedding_model = None
_embedding_model_lock = threading.Lock()
_embedding_encode_lock = threading.Lock()


def _get_driver():
    global _driver
    if _driver is not None:
        return _driver
    with _driver_lock:
        if _driver is not None:
            return _driver
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            raise RuntimeError("缺少 NEO4J_PASSWORD，請在匯入專案模組前載入環境設定。")
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), password),
            connection_timeout=10,
        )
        try:
            driver.verify_connectivity()
        except Exception:
            driver.close()
            raise
        _driver = driver
        logger.info("Neo4j 已連線：database=%s", NEO4J_DATABASE)
        return driver


def close_neo4j_driver():
    """於工作佇列停止且查詢結束後呼叫。"""
    global _driver
    with _driver_lock:
        driver, _driver = _driver, None
        if driver is not None:
            driver.close()


def _read(query: str, parameters: dict | None = None) -> list[dict]:
    @unit_of_work(timeout=NEO4J_QUERY_TIMEOUT)
    def execute(tx):
        return [dict(record) for record in tx.run(query, parameters or {})]
    with _get_driver().session(
        database=NEO4J_DATABASE, default_access_mode=READ_ACCESS
    ) as session:
        return session.execute_read(execute)


def _unique_strings(values: Any) -> list[str]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]
    return list(dict.fromkeys(
        value.strip() for value in values
        if isinstance(value, str) and value.strip()
    ))


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        number = default
    return max(minimum, min(number, maximum))


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().replace("台", "臺")


def _empty_spot(spot_key: str) -> dict:
    return _normalize_location({"name": spot_key})


def _normalize_location(record: dict) -> dict:
    location = dict(record)

    uid = str(
        location.get("uid")
        or location.get("uuid")
        or location.get("id")
        or ""
    ).strip()

    location.update(
        {
            "uid": uid,
            "id": uid,
            "uuid": uid,
        }
    )

    for key in (
        "name",
        "description",
        "address",
        "type",
    ):
        location[key] = str(
            location.get(key) or ""
        ).strip()

    for key in (
        "labels",
        "images",
        "tags",
        "categories",
        "cuisines",
        "cities",
        "towns",
        "town_ids",
        "matched_keywords",
    ):
        location[key] = _unique_strings(
            location.get(key)
        )

    location["tags"] = _unique_strings(
        location["tags"] + location["categories"]
    )

    raw_hours = location.get("operating_hours")

    if raw_hours is None:
        raw_hours = []

    if not isinstance(raw_hours, list):
        raise ValueError(
            "operating_hours 必須是清單"
        )

    operating_hours = []
    seen_hours = set()

    for entry in raw_hours:
        if not isinstance(entry, dict):
            raise ValueError(
                "operating_hours 每筆必須是物件"
            )

        normalized = {
            "dayOfWeek": str(
                entry.get("dayOfWeek") or ""
            ).strip(),
            "openTime": str(
                entry.get("openTime") or ""
            ).strip(),
            "closeTime": str(
                entry.get("closeTime") or ""
            ).strip(),
        }

        # 只排除完全空白的紀錄。
        # 部分缺值仍保留，避免把不完整資料誤當不存在。
        if not any(normalized.values()):
            continue

        identity = (
            normalized["dayOfWeek"],
            normalized["openTime"],
            normalized["closeTime"],
        )

        if identity in seen_hours:
            continue

        seen_hours.add(identity)
        operating_hours.append(normalized)

    location["operating_hours"] = operating_hours
    location["has_operating_hours"] = bool(
        operating_hours
    )

    location.setdefault("lat", None)
    location.setdefault("lon", None)

    return location

# 只選實體地點，排除掛有 Place 標籤的版本節點。
_ENTITY = """
NOT p:Version AND p.uid IS NOT NULL AND
(p:Attraction OR p:Restaurant OR p:Hotel OR p:Event OR p:MerchantPlace)
"""

_TYPE = """
CASE
    WHEN p:Attraction THEN 'Attraction'
    WHEN p:Restaurant THEN 'Restaurant'
    WHEN p:Hotel THEN 'Hotel'
    WHEN p:Event THEN 'Event'
    WHEN p:MerchantPlace THEN 'MerchantPlace'
    ELSE 'Place'
END
"""

_NAME = """
CASE
    WHEN trim(coalesce(v.name, '')) <> '' THEN v.name
    WHEN trim(coalesce(p.name, '')) <> '' THEN p.name
    ELSE coalesce(p.EventName, '')
END
"""

_DESCRIPTION = """
CASE
    WHEN trim(coalesce(v.description, '')) <> '' THEN v.description
    WHEN trim(coalesce(p.description, '')) <> '' THEN p.description
    ELSE coalesce(p.Description, '')
END
"""

_ADDRESS = """
CASE
    WHEN trim(coalesce(v.address, '')) <> '' THEN v.address
    ELSE coalesce(p.address, p.Address, '')
END
"""

_GEO_FILTER = """
(
    $city_name = ''
    OR EXISTS {
        MATCH (p)-[:LOCATED_IN_CITY]->(c:City)
        WHERE replace(c.name, '台', '臺') = $city_name
    }
    OR EXISTS {
        MATCH (p)-[:LOCATED_IN_TOWN]->(:Town)-[:PART_OF]->(c:City)
        WHERE replace(c.name, '台', '臺') = $city_name
    }
)
AND
(
    $town_name = ''
    OR EXISTS {
        MATCH (p)-[:LOCATED_IN_TOWN]->(t:Town)
        WHERE replace(coalesce(t.name, ''), '台', '臺') = $town_name
           OR replace(coalesce(t.id, ''), '台', '臺') = $town_name
    }
)
"""

_CURRENT_VERSION = """
OPTIONAL MATCH (p)-[:HAS_VERSION]->(candidate:Version:Current)
WITH p, candidate
ORDER BY candidate.version_no DESC
WITH p, head(collect(candidate)) AS v
"""

_DETAILS = f"""
p.uid AS uid,
p.uid AS uuid,
p.uid AS id,
{_TYPE} AS type,
labels(p) AS labels,
{_NAME} AS name,
{_DESCRIPTION} AS description,
{_ADDRESS} AS address,
coalesce(p.lat, p.PositionLat) AS lat,
coalesce(p.lon, p.PositionLon) AS lon,
p.ticket_info AS ticket_info,
p.business_status AS business_status,
p.service_status AS service_status,
[(p)-[:HAS_TAG]->(t:Tag) | t.name] AS tags,
[(p)-[:HAS_CATEGORY]->(c:Category) | c.name] AS categories,
CASE
    WHEN p:Restaurant THEN
        [(p)-[:HAS_CATEGORY]->(c:Category) | c.name]
    ELSE []
END AS cuisines,
[(p)-[:LOCATED_IN_CITY]->(c:City) | c.name] +
[(p)-[:LOCATED_IN_TOWN]->(:Town)-[:PART_OF]->(c:City) | c.name] AS cities,
[(p)-[:LOCATED_IN_TOWN]->(t:Town) | t.name] AS towns,
[(p)-[:LOCATED_IN_TOWN]->(t:Town) | t.id] AS town_ids,
[(p)-[:HAS_IMAGE]->(i:Image)
 WHERE coalesce(i.rehost_status, '') <> 'unavailable'
 | i.url] AS images,
v.image_url AS version_image_url,
[(p)-[:HAS_OPERATING_HOURS]->(h:OperatingHours) |
    {{
        dayOfWeek: h.dayOfWeek,
        openTime: h.openTime,
        closeTime: h.closeTime
    }}
] AS operating_hours
"""

def execute_readonly_cypher(query: str, parameters: dict | None = None):
    """預設停用；字串過濾並非安全沙箱，必須另配唯讀帳號與 API 授權。"""
    if os.getenv("NEO4J_ALLOW_RAW_READ_QUERY", "0").lower() not in {"1", "true", "yes"}:
        raise PermissionError("自訂 Cypher 預設停用，請先配置唯讀帳號與 API 授權。")
    if not isinstance(query, str) or not query.strip() or len(query) > 20000:
        raise ValueError("query 必須是非空字串，最多 20000 字元。")
    statement = query.strip()
    if not re.match(r"^(?:MATCH|OPTIONAL\s+MATCH)\b", statement, re.I):
        raise ValueError("僅接受 MATCH 或 OPTIONAL MATCH 開頭的查詢。")
    if re.search(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|ALTER|RENAME|GRANT|DENY|REVOKE|LOAD|FOREACH|CALL)\b", statement, re.I):
        raise ValueError("查詢包含不允許的關鍵字。")
    if any(marker in statement for marker in (";", "//", "/*", "*/")):
        raise ValueError("不允許分號與查詢註解。")
    return _read(statement, parameters)


def _build_description_keywords(preferences: list[str] | None) -> list[str]:
    prefixes = ("我們想要", "我想要", "我要", "想要", "希望", "想去", "想看", "想吃", "想", "去", "看", "吃")
    result = []
    for original in _unique_strings(preferences):
        cleaned = original
        for _ in range(6):
            previous = cleaned
            for prefix in prefixes:
                if cleaned.startswith(prefix) and len(cleaned[len(prefix):].strip()) >= 2:
                    cleaned = cleaned[len(prefix):].strip()
                    break
            if previous == cleaned:
                break
        for word in (original, cleaned):
            if len(word) >= 2 and word not in result:
                result.append(word)
    return result[:16]


def _query_locations(*, labels: list[str], keywords: list[str], city_name: str,
                     town_name: str, limit: int, require_keywords: bool) -> list[dict]:
    query = f"""
    MATCH (p:Place)
    WHERE {_ENTITY}
      AND any(label IN labels(p) WHERE label IN $labels)
      AND {_GEO_FILTER}
    {_CURRENT_VERSION}
    WITH p, v, {_NAME} AS name, {_DESCRIPTION} AS description
    WHERE trim(name) <> '' AND trim(description) <> ''
    WITH p, v, name, description, [k IN $keywords WHERE
        toLower(name) CONTAINS toLower(k)
        OR toLower(description) CONTAINS toLower(k)
        OR any(a IN coalesce(p.alt_names, []) WHERE toLower(a) CONTAINS toLower(k))
        OR EXISTS {{ MATCH (p)-[:HAS_TAG|HAS_CATEGORY]->(dimension)
                     WHERE toLower(coalesce(dimension.name, '')) CONTAINS toLower(k) }}
    ] AS matched_keywords
    WHERE NOT $require_keywords OR size(matched_keywords) > 0
    WITH p, v, matched_keywords, size(matched_keywords) AS keyword_score
    ORDER BY keyword_score DESC, p.uid ASC LIMIT $limit
    RETURN {_DETAILS}, matched_keywords, keyword_score
    ORDER BY keyword_score DESC, uuid ASC
    """
    records = _read(query, {
        "labels": labels, "keywords": keywords,
        "city_name": _normalize_text(city_name), "town_name": _normalize_text(town_name),
        "limit": _bounded_int(limit, 4, 1, 50), "require_keywords": require_keywords,
    })
    return [_normalize_location(record) for record in records]


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_model_lock:
        if _embedding_model is not None:
            return _embedding_model
        from sentence_transformers import SentenceTransformer
        device = os.getenv("EMBEDDING_DEVICE", "cpu").strip() or "cpu"
        print(f"[Embedding] 載入 {E5_MODEL_NAME}｜裝置={device}")
        model = SentenceTransformer(E5_MODEL_NAME, device=device)
        if model.get_sentence_embedding_dimension() != E5_DIMENSION:
            raise RuntimeError("Embedding 模型維度與資料庫預期的 1024 維不一致。")
        _embedding_model = model
        print("[Embedding] 模型載入完成")
    return _embedding_model


def _encode_search_query(text: str) -> list[float]:
    text = str(text or '').strip()
    if not text:
        raise ValueError("語意檢索文字不可為空白。")
    model = _get_embedding_model()
    with _embedding_encode_lock:
        vector = model.encode(
            "query: " + text, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
    values = vector.tolist()
    if len(values) != E5_DIMENSION or not all(math.isfinite(v) for v in values):
        raise ValueError("問句向量維度或數值無效。")
    if not any(v != 0 for v in values):
        raise ValueError("問句向量不可全部為零。")
    return values


def recommend_locations_for_script(
    user_prompt: str,
    preferences: list[str] | None = None,
    city_name: str | None = None,
    town_name: str | None = None,
    limit: int = 4,
    is_night: bool = False,
) -> list:
    """
    E5 語意候選檢索。

    指定卡皮巴拉／水豚時，先限制為資料中提及相關名稱的候選，
    再進行向量排序。

    文字命中不等於已完成需求、營業或現場可觀賞狀態驗證。
    此函式不建立索引、不寫入資料。
    """
    preferences = _unique_strings(preferences)

    search_text = (
        "；".join(preferences)
        if preferences
        else str(user_prompt or "").strip()
    )

    if not search_text:
        raise ValueError("必須提供 user_prompt 或 preferences。")

    # 這裡只針對目前明確的指定動物需求。
    # 若 preferences 已存在，以解析出的正向需求為準，
    # 不再從整句 user_prompt 任意補入其他條件。
    normalized_search = search_text.casefold()

    capybara_terms = [
        "卡皮巴拉",
        "水豚",
        "capybara",
    ]

    required_terms = []

    if any(
        term in normalized_search
        for term in capybara_terms
    ):
        required_terms = capybara_terms

    print(
        f"[Neo4j 檢索條件]"
        f"｜需求={search_text!r}"
        f"｜指定對象關鍵詞={required_terms}"
    )

    vector = _encode_search_query(search_text)

    query = f"""
    MATCH (p:Place:Attraction)
    WHERE NOT p:Version
      AND p.uid IS NOT NULL
      AND p.description_embedding IS NOT NULL
      AND size(p.description_embedding) = $dimension
      AND {_GEO_FILTER}

    {_CURRENT_VERSION}

    WITH p, v,
         {_NAME} AS name,
         {_DESCRIPTION} AS description

    WHERE trim(name) <> ''
      AND trim(description) <> ''

    WITH p, v, name, description,
         [term IN $required_terms WHERE
             toLower(name) CONTAINS term
             OR toLower(description) CONTAINS term
             OR any(
                 alias IN coalesce(p.alt_names, [])
                 WHERE toLower(alias) CONTAINS term
             )
             OR EXISTS {{
                 MATCH (p)-[:HAS_TAG|HAS_CATEGORY]->(dimension)
                 WHERE toLower(
                     coalesce(dimension.name, '')
                 ) CONTAINS term
             }}
         ] AS matched_keywords

    WHERE size($required_terms) = 0
       OR size(matched_keywords) > 0

    WITH p, v, matched_keywords,
         vector.similarity.cosine(
             p.description_embedding,
             $vector
         ) AS semantic_score

    WHERE semantic_score IS NOT NULL

    ORDER BY semantic_score DESC, p.uid ASC
    LIMIT $limit

    RETURN
        {_DETAILS},
        semantic_score,
        matched_keywords

    ORDER BY semantic_score DESC, uuid ASC
    """

    records = _read(
        query,
        {
            "vector": vector,
            "dimension": E5_DIMENSION,
            "city_name": _normalize_text(city_name),
            "town_name": _normalize_text(town_name),
            "limit": _bounded_int(limit, 4, 1, 8),
            "required_terms": required_terms,
        },
    )

    if required_terms and not records:
        raise ValueError(
            "目前資料庫中，在指定地區與檢索條件下，"
            "找不到文字資料提及卡皮巴拉或水豚、"
            "且具有可用介紹與向量的景點；"
            "未生成其他地點替代。"
        )

    locations = []

    for record in records:
        location = _normalize_location(record)

        location.update(
            {
                "retrieval_mode": (
                    "e5_cosine_with_entity_keyword_filter"
                    if required_terms
                    else "e5_exact_cosine"
                ),
                "search_text": search_text,
                "required_entity_terms": list(required_terms),
                "entity_keyword_filter_applied": bool(
                    required_terms
                ),
                # 關鍵字命中不是完整語意或即時狀態驗證。
                "requires_requirement_review": True,
                "night_requested": bool(is_night),
                "night_opening_verified": False,
                "route_feasibility_verified": False,
                "business_status_verified": False,
            }
        )

        locations.append(location)

    print(
        f"[Neo4j 語意檢索]"
        f" 文字={search_text!r}"
        f"｜候選數={len(locations)}"
        f"｜指定對象篩選={bool(required_terms)}"
    )

    for index, location in enumerate(locations, start=1):
        print(
            f"[語意候選 {index}]"
            f" {location['name']}"
            f"｜相似度={location['semantic_score']:.4f}"
            f"｜命中={location['matched_keywords']}"
            f"｜縣市={location['cities']}"
            f"｜行政區={location['towns']}"
        )

    return locations


def fetch_locations_for_script(town_name: str, limit: int = 4, is_night: bool = False,
                               *, city_name: str | None = None,
                               preferences: list[str] | None = None, user_prompt: str = '') -> list:
    """相容舊介面；有需求時語意檢索，否則只回傳行政區候選。"""
    if str(user_prompt or '').strip() or _unique_strings(preferences):
        return recommend_locations_for_script(user_prompt, preferences, city_name, town_name, limit, is_night)
    if not str(town_name or '').strip():
        raise ValueError("未提供旅遊需求時，必須指定 town_name。")
    locations = _query_locations(labels=['Attraction'], keywords=[], city_name=city_name or '',
                                 town_name=town_name, limit=_bounded_int(limit, 4, 1, 8), require_keywords=False)
    for location in locations:
        location.update({'retrieval_mode': 'administrative_candidates',
                         'preference_filter_applied': False, 'night_requested': bool(is_night),
                         'night_opening_verified': False, 'route_feasibility_verified': False})
    return locations


def fetch_spot_complete_info(spot_key: str) -> dict:
    """精確 uid 優先，其次名稱與 alt_names；商家讀取 Current 版本。

    id/uuid 回傳欄位皆為 uid 的相容別名，不使用已移除的舊來源 ID。
    """
    spot_key = str(spot_key or '').strip()
    if not spot_key:
        return _empty_spot('')
    query = f"""
    MATCH (p:Place) WHERE {_ENTITY}
    {_CURRENT_VERSION}
    WITH p, v, {_NAME} AS name
    WHERE p.uid = $key OR name = $key OR p.name = $key OR p.EventName = $key
       OR $key IN coalesce(p.alt_names, [])
    WITH p, v, CASE WHEN p.uid = $key THEN 0 ELSE 1 END AS match_priority
    ORDER BY match_priority, p.uid LIMIT 2
    RETURN {_DETAILS}, match_priority
    ORDER BY match_priority, uuid
    """
    records = _read(query, {'key': spot_key})
    if not records:
        return _empty_spot(spot_key)
    if len(records) > 1 and records[0]['match_priority'] == records[1]['match_priority']:
        raise ValueError(f"「{spot_key}」符合多個地點，請使用唯一 uid。")
    record = records[0]
    record.pop('match_priority', None)
    return _normalize_location(record)


def search_neo4j_rag(keyword: str, **kwargs) -> str:
    """相容舊介面的關鍵字背景查詢；不是向量搜尋。"""
    keyword = str(keyword or '').strip()
    if not keyword:
        return '未提供檢索關鍵字。'
    locations = _query_locations(
        labels=['Attraction', 'Restaurant', 'Hotel', 'Event', 'MerchantPlace'],
        keywords=_build_description_keywords([keyword]) or [keyword],
        city_name=str(kwargs.get('city_name') or ''), town_name=str(kwargs.get('town_name') or ''),
        limit=_bounded_int(kwargs.get('limit'), 3, 1, 10), require_keywords=True,
    )
    if not locations:
        return '查無符合關鍵字的在地知識。'
    return '\n\n---\n\n'.join(
        f"地點識別碼：{p['uid']}\n地點：{p['name']}\n地址：{p['address'] or '未提供'}\n介紹：{p['description'][:1000]}"
        for p in locations
    )


def find_spots_by_intent(lat: float, lon: float, tags: list,
                         max_distance_meters: int = 2000) -> list:
    """附近 Attraction；tags 為名稱／描述／標籤的 OR 關鍵字篩選。"""
    try:
        lat, lon, radius = float(lat), float(lon), float(max_distance_meters)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('經緯度與距離必須是有效數值。') from exc
    if not all(math.isfinite(v) for v in (lat, lon, radius)):
        raise ValueError('經緯度與距離不可為無限值或 NaN。')
    if not (-90 <= lat <= 90 and -180 <= lon <= 180 and radius > 0):
        raise ValueError('經緯度超出範圍，或距離不大於零。')
    query = """
    WITH point({latitude: $lat, longitude: $lon}) AS here
    MATCH (p:Place:Attraction) WHERE NOT p:Version AND p.uid IS NOT NULL
    WITH p, here, toFloat(p.lat) AS lat, toFloat(p.lon) AS lon
    WITH p, here, CASE
        WHEN p.location IS NOT NULL THEN p.location
        WHEN lat >= -90 AND lat <= 90 AND lon >= -180 AND lon <= 180
        THEN point({latitude: lat, longitude: lon}) ELSE null END AS position
    WHERE position IS NOT NULL
    WITH p, point.distance(position, here) AS distance_m
    WHERE distance_m <= $radius
    WITH p, distance_m, [k IN $keywords WHERE
        toLower(coalesce(p.name, '')) CONTAINS toLower(k)
        OR toLower(coalesce(p.description, '')) CONTAINS toLower(k)
        OR EXISTS { MATCH (p)-[:HAS_TAG]->(tag:Tag)
                    WHERE toLower(coalesce(tag.name, '')) CONTAINS toLower(k) }
    ] AS matched_keywords
    WHERE size($keywords) = 0 OR size(matched_keywords) > 0
    RETURN p.uid AS uuid, p.uid AS id, 'Attraction' AS type,
        p.name AS name, coalesce(p.description, '') AS description,
        coalesce(p.address, '') AS address, p.lat AS lat, p.lon AS lon,
        [(p)-[:HAS_TAG]->(t:Tag) | t.name] AS tags,
        matched_keywords, toInteger(distance_m) AS distance_m
    ORDER BY size(matched_keywords) DESC, distance_m, uuid LIMIT 3
    """
    return [_normalize_location(r) for r in _read(query, {
        'lat': lat, 'lon': lon, 'radius': radius,
        'keywords': _build_description_keywords(tags),
    })]
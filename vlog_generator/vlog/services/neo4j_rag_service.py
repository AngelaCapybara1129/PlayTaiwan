import os
from neo4j import GraphDatabase

_driver = None
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "playtaiwandb")  # 指定要使用的資料庫

def _get_driver():
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "playtaiwan2026")
        try:
            _driver = GraphDatabase.driver(uri, auth=(user, password))
            _driver.verify_connectivity()
            print("✅ 成功連線至真實 Neo4j 圖形資料庫！")
        except Exception as e:
            print(f"❌ Neo4j 連線失敗: {e}")
            _driver = None
    return _driver

def execute_readonly_cypher(query: str, parameters: dict = None):
    driver = _get_driver()
    if not driver:
        raise ConnectionError("Neo4j 資料庫未連線。")

    if not query.strip().upper().startswith("MATCH"):
        raise ValueError("安全性限制：僅允許執行 MATCH 查詢語法。")

    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(query, parameters or {})
        return [dict(record) for record in result]

def search_neo4j_rag(keyword: str, **kwargs) -> str:
    """
    根據關鍵字檢索 Neo4j 中的在地知識。
    使用 **kwargs 接收其他未使用的參數（如 is_night_mode 等），避免多餘參數傳入時報錯。
    """
    driver = _get_driver()
    if not driver:
        return "Neo4j 未連線，無法提供在地背景知識。"

    query = """
    MATCH (n)
    WHERE n.name CONTAINS $keyword OR n.EventName CONTAINS $keyword
    RETURN COALESCE(n.name, n.EventName) AS name,
           COALESCE(n.description, n.Description, "暫無介紹") AS desc,
           COALESCE(n.address, n.Address, "無地址") AS addr
    LIMIT 3
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, keyword=keyword)
            contexts = []
            for record in result:
                contexts.append(f"地點: {record['name']}\n地址: {record['addr']}\n介紹: {record['desc'][:200]}...")
            return "\n---\n".join(contexts) if contexts else "查無相關在地知識。"
    except Exception as e:
        return f"RAG 檢索發生錯誤: {e}"

def fetch_spot_complete_info(spot_key: str) -> dict:
    """
    【嚴格防幻覺查詢 + 回憶素材擴充】
    精準從 Neo4j 提取指定地點的真實資料，支援透過 uid 或名稱檢索。

    🆕 修正：新版 mergedb 主鍵已從 `id` 統一改為 `uid`，
    舊查詢用 n.id 比對會永遠比對不到資料，故改用 n.uid。
    """
    driver = _get_driver()
    if not driver:
        return {"id": "", "name": spot_key, "description": "", "address": "", "images": [], "tags": []}

    query = """
    MATCH (n)
    WHERE n.uid = $spot_key OR n.name = $spot_key OR n.EventName = $spot_key
    OPTIONAL MATCH (n)-[:HAS_IMAGE]->(img:Image)
    OPTIONAL MATCH (n)-[:HAS_TAG]->(t:Tag)
    OPTIONAL MATCH (n)-[:HAS_CATEGORY]->(cat:Category)
    RETURN COALESCE(n.uid, elementId(n)) AS id,
           labels(n)[0] AS type,
           COALESCE(n.name, n.EventName) AS name,
           COALESCE(n.description, n.Description, "") AS description,
           COALESCE(n.address, n.Address, "") AS address,
           collect(DISTINCT img.url) AS images,
           [x IN collect(DISTINCT t.name) WHERE x IS NOT NULL] + [x IN collect(DISTINCT cat.name) WHERE x IS NOT NULL] AS tags
    LIMIT 1
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, spot_key=spot_key)
            record = result.single()
            if record and record["description"]:
                return {
                    "id": record["id"],
                    "name": record["name"],
                    "type": record["type"],
                    "description": record["description"],
                    "address": record["address"],
                    "images": [img for img in record["images"] if img],
                    "tags": record["tags"]
                }
    except Exception as e:
        print(f"⚠️ Neo4j 完整資訊檢索錯誤: {e}")

    return {"id": "", "name": spot_key, "description": "", "address": "", "images": [], "tags": []}


def fetch_locations_for_script(town_name: str, limit: int = 4, is_night: bool = False) -> list:
    """
    動態旅遊動線規劃：
    - 白天模式：依據 limit 動態分配 [景點, 餐廳, 景點, 旅宿]
    - 夜間模式：優先撈取夜間活動、餐廳或旅宿相關節點

    🆕 修正：Town 唯一鍵已從 name 改為 id（格式：縣市名_鄉鎮名），
    且同名鄉鎮可能跨縣市重複，故比對條件加入：
      1. t.name 精確比對
      2. t.id 結尾比對（同時涵蓋「縣市_鄉鎮」格式）
      3. 「台/臺」混用容錯（用 replace 統一成「台」再比對）
    """
    driver = _get_driver()
    if not driver:
        raise ConnectionError("Neo4j 資料庫未連線，無法撈取實體景點。")

    query_template = """
    MATCH (place:Place)-[:LOCATED_IN_TOWN]->(t:Town)
    WHERE (
        t.name = $town_name
        OR t.id ENDS WITH $town_name
        OR replace(t.name, '臺', '台') = replace($town_name, '臺', '台')
    )
    AND place:%s
    OPTIONAL MATCH (place)-[:HAS_TAG]->(tag:Tag)
    OPTIONAL MATCH (place)-[:HAS_CATEGORY]->(cat:Category)
    RETURN place.uid AS uuid,
           labels(place)[0] AS type,
           COALESCE(place.name, "未知名稱") AS name,
           COALESCE(place.description, "暫無介紹") AS description,
           COALESCE(place.address, place.Address, "無地址") AS address,
           place.ticket_info AS ticket_info,
           collect(DISTINCT tag.name) AS tags,
           collect(DISTINCT cat.name) AS categories
    ORDER BY rand()
    LIMIT 1
    """

    target_labels = []

    if is_night:
        target_labels = ["Restaurant", "Attraction", "Hotel"]
        while len(target_labels) < limit:
            target_labels.insert(1, "Attraction")
        target_labels = target_labels[:limit]
    else:
        if limit == 1:
            target_labels = ["Attraction"]
        elif limit == 2:
            target_labels = ["Attraction", "Restaurant"]
        elif limit == 3:
            target_labels = ["Attraction", "Restaurant", "Hotel"]
        else:
            target_labels.append("Attraction")
            middle_count = limit - 2
            restaurant_count = max(1, middle_count // 2)
            attraction_count = middle_count - restaurant_count
            for _ in range(attraction_count):
                target_labels.append("Attraction")
            for _ in range(restaurant_count):
                target_labels.append("Restaurant")
            target_labels.append("Hotel")

    while len(target_labels) < limit:
        target_labels.insert(1, "Attraction")
    target_labels = target_labels[:limit]

    combined_locations = []
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            for label in target_labels:
                current_query = query_template % label
                result = session.run(current_query, town_name=town_name)
                record = result.single()

                if not record:
                    fb_result = session.run(query_template % "Attraction", town_name=town_name)
                    record = fb_result.single()

                if record:
                    loc_dict = dict(record)
                    if loc_dict['type'] == 'Restaurant':
                        loc_dict['cuisines'] = loc_dict.pop('categories', [])
                    else:
                        loc_dict['tags'] = loc_dict.get('tags', []) + loc_dict.pop('categories', [])
                        loc_dict['cuisines'] = []

                    combined_locations.append(loc_dict)

        print(f"📍 [Neo4j] 為 {town_name} ('夜間模式'={is_night}) 規劃了 {len(combined_locations)} 站動線: {target_labels}")
        return combined_locations
    except Exception as e:
        print(f"⚠️ 動態獲取劇本地點失敗: {e}")
        return []


def find_spots_by_intent(lat: float, lon: float, tags: list, max_distance_meters: int = 2000) -> list:
    """
    Step 3: Graph RAG (知識圖譜過濾)
    根據 Agent 解析出的標籤與當下 GPS 位置進行空間與屬性檢索。

    ⚠️ 注意：僅查詢 Attraction（空間資料覆蓋率 97.8%）。
    Restaurant 的 location (Point) 覆蓋率僅 0.6%，若未來要納入餐廳，
    空間距離篩選在餐廳資料上幾乎無效，需改用其他方式（如同 Town/City 篩選）。
    """
    driver = _get_driver()
    if not driver:
        return []

    query = """
    WITH point({srid: 4326, x: $lon, y: $lat}) AS UserLocation
    MATCH (a:Attraction)
    WHERE a.location IS NOT NULL
      AND point.distance(a.location, UserLocation) <= $max_distance

    OPTIONAL MATCH (a)-[:HAS_TAG]->(t:Tag)
    WITH a, UserLocation, collect(t.name) AS spot_tags

    RETURN a.name AS name,
           a.address AS address,
           a.description AS description,
           spot_tags AS tags,
           toInteger(point.distance(a.location, UserLocation)) AS distance_m
    ORDER BY distance_m ASC
    LIMIT 3
    """

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, lat=lat, lon=lon, max_distance=max_distance_meters)
            return [dict(r) for r in result]
    except Exception as e:
        print(f"⚠️ Agent 檢索圖譜失敗: {e}")
        return []
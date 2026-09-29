import re
import time
import json
import random
import threading
import urllib.parse
from datetime import datetime, timedelta, timezone

import torch
from pydantic import ValidationError
from api.schemas import (
    DialogueInput, DialogueOutput,
    OvernightTransitionInput, OvernightTransitionOutput,
    NarrationInput, NarrationOutputNode
)
from transformers import pipeline
from core.config import TEMPLATES

# ============================================================
# LLM 載入與執行緒安全
# ============================================================
_llm_pipeline = None
_llm_wrapper = None
_load_lock = threading.Lock()   # 避免多個請求同時載入模型
_gen_lock = threading.Lock()    # 同一時間只讓一個請求使用 GPU 生成


class _LockedLLM:
    """包一層鎖與計時，所有 llm(...) 呼叫點不用改"""

    def __init__(self, pipe):
        self._pipe = pipe

    def __call__(self, *args, **kwargs):
        with _gen_lock:
            t0 = time.time()
            out = self._pipe(*args, **kwargs)
            print(f"⏱️ LLM 生成耗時 {time.time() - t0:.1f}s")
            return out


def _get_llm():
    global _llm_pipeline, _llm_wrapper
    if _llm_wrapper is None:
        with _load_lock:
            if _llm_wrapper is None:
                print("🚀 正在載入 LLM 大腦 (Llama-3-8B-Instruct)...")
                _llm_pipeline = pipeline(
                    "text-generation",
                    model="NousResearch/Meta-Llama-3-8B-Instruct",
                    dtype=torch.bfloat16,  # 舊版 transformers 不認得 dtype 時，改回 model_kwargs={"torch_dtype": torch.bfloat16}
                    device_map="auto",
                )
                print(f"✅ LLM 已載入，裝置: {_llm_pipeline.model.device}")
                _llm_wrapper = _LockedLLM(_llm_pipeline)
    return _llm_wrapper


def preload_llm():
    """伺服器啟動時呼叫，避免第一個請求才載入模型"""
    _get_llm()


# ============================================================
# 1. Vlog 旁白生成
# ============================================================
def generate_vlog_content_with_template(
    raw_text: str,
    emotion: str,
    template: str,
    rag_context: str = "",
    play_time: str = "",
    game_tasks: str = "",
    promo_info: str = ""
) -> dict:
    llm = _get_llm()
    tpl = TEMPLATES.get(template, TEMPLATES.get("user_vlog", {"name": "一般生活紀錄", "tone": "自然"}))

    extra_info = ""
    if "user" in template or "visitor" in template:
        extra_info += f"【遊玩時間】：{play_time}\n" if play_time else ""
        extra_info += f"【完成任務】：{game_tasks}\n" if game_tasks else ""
    elif "merchant" in template or "promo" in template:
        extra_info += f"【商家優惠/主打】：{promo_info}\n" if promo_info else ""

    # 專為「玩家回憶紀錄 Vlog」打造的系統提示詞
    system_prompt = f"""你是一位擅長撰寫動人旅遊日記的編劇與社群行銷達人。
【當前任務】：為玩家生成專屬的「實境解謎回憶 Vlog」旁白與社群貼文。
【在地記憶 (RAG & 關卡)】：\n{rag_context}
【玩家狀態】：{extra_info}
【情緒氛圍】：{emotion}

【嚴格撰寫守則】：
1. 視角與口吻：旁白 (tw_script) 必須以「玩家第一人稱 (我/我們)」撰寫，宛如在翻閱剛寫完的旅行日記，語氣要自然、真誠。
2. 內容融合：必須將「在地歷史文化知識」、「與當地人(NPC)的互動解謎」以及「收集到的過關明信片」自然地揉合在旁白中。不要死板地背書，要展現出親身經歷的感悟。
3. 社群宣傳：promo_copy 要非常適合發佈在 Instagram、小紅書或 Facebook，帶有滿滿的成就感、適當的 Emoji，並附上相關的 Hashtag (如 #實境解謎 #旅遊紀錄 #專屬明信片 等)。
4. 語言限制：除了 en_video_prompt 必須為英文外，其餘所有內容**必須 100% 使用繁體中文 (zh-TW)**，嚴禁夾雜英文單字或簡體字！
5. 嚴禁在旁白或文案中提及「資料庫」、「Neo4j」、「RAG」、「系統」、「AI」等技術名詞，一切都要像真實的旅行回憶。

請務必輸出以下完整合法的 JSON 格式：
{{
    "tw_script": "Vlog的繁體中文旁白腳本 (約 150-200 字，第一人稱，包含文化感悟、任務互動與明信片回憶)",
    "en_video_prompt": "給AI影片生成引擎的英文場景提示詞，詳細描述畫面視覺",
    "seo_keywords": ["繁體中文關鍵字1", "繁體中文關鍵字2"],
    "target_audience": ["繁體中文客群1", "繁體中文客群2"],
    "promo_copy": "適合發布在 IG/FB 的繁體中文宣傳貼文文案 (包含 Emoji 與 Hashtag)"
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"使用者提供素材/要求：{raw_text}"}
    ]

    # 旁白約 200 字 + 文案，1000 tokens 已足夠，也比較不會拖慢回應
    outputs = llm(messages, max_new_tokens=1000, temperature=0.7, do_sample=True)

    response = outputs[0]["generated_text"][-1]["content"]

    # JSON 擷取與自動修復防呆機制
    start = response.find("{")
    end = response.rfind("}")

    try:
        if start != -1:
            if end == -1 or end < start:
                # 結尾的 } 被截斷了，幫它補齊
                json_str = response[start:] + "\n}"
            else:
                json_str = response[start:end + 1]
            return json.loads(json_str)
        else:
            raise ValueError("找不到 JSON 的開頭大括號")

    except Exception as e:
        print(f"❌ LLM JSON 解析失敗: {e}\n原始輸出: {response}")
        return {
            "tw_script": "歡迎來到這段美好的旅程，讓我們一起探索在地文化與獨特風情！",
            "en_video_prompt": "beautiful scenic landscape, high quality, cinematic lighting",
            "seo_keywords": ["台灣旅遊", "在地體驗", "特色觀光"],
            "target_audience": ["旅遊愛好者", "大眾客群"],
            "promo_copy": "展開一段美好的旅程！快來這裡探索未知的驚喜吧！ #旅遊 #探索"
        }


def _parse_llm_json(response_text: str):
    start, end = response_text.find("{"), response_text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("找不到 JSON 結構")
    return json.loads(response_text[start:end])


# ============================================================
# 2. 遊戲劇情節點生成
# ============================================================
def generate_story_node(payload_dict: dict) -> dict:
    llm = _get_llm()
    node_type = payload_dict.get("node_type")
    max_retries = 3

    if node_type == "dialogue":
        req = DialogueInput(**payload_dict)
        npc = req.npcs[0] if req.npcs else type("Dummy", (), {"npc_id": "npc_01", "name": "神秘引導員", "role": "NPC"})()

        sys_prompt = f"""你是一個實境遊戲的劇情 AI 引擎。請務必嚴格輸出一個純 JSON 物件（不要包含任何 Markdown 標記或額外文字），用來回應玩家的對話與推進劇情。

【絕對語言指令】：
請務必 100% 使用台灣慣用的「繁體中文 (zh-TW)」來生成 NPC 的台詞與旁白，絕對嚴禁出現簡體字或夾雜任何英文單字！台詞要自然、生動，符合在地人的說話口吻。

【當前地點】：{req.location.name} ({req.location.address}) - {req.location.description}
【當前目標】：{req.node_context.goal}
【場景描述】：{req.node_context.scene_description}
【扮演 NPC】：{npc.name} ({getattr(npc, 'role', 'NPC')})。
【玩家輸入】：{req.player_input}

【輸出 JSON 結構規範（請務必包含以下所有欄位，型態需完全相符）】：
{{
  "location_id": "{req.location.location_id}",
  "node_id": "{req.node_id}",
  "narration": {{
    "opening_hook": "吸引玩家的短旁白懸念 (繁體中文)",
    "scene_description": "當前場景的進一步氛圍描述 (繁體中文)",
    "historical_note": null
  }},
  "npc_dialogue": [
    {{
      "npc_id": "{getattr(npc, 'npc_id', 'npc_01')}",
      "line": "NPC 回應玩家的台詞 (務必使用純繁體中文)",
      "emotion": "happy",
      "handoff_to": null
    }}
  ],
  "player_choices": [
    {{
      "choice_id": "choice_1",
      "text": "引導玩家繼續下一步的選項按鈕文字 (繁體中文)"
    }}
  ]
}}
*(注意：emotion 欄位的值僅限：happy, neutral, angry, sad, excited 之一)*
"""
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"玩家剛才說：「{req.player_input}」。請根據上述規範產生對應的 JSON 回應。"}
        ]

        for attempt in range(max_retries):
            try:
                outputs = llm(
                    msgs,
                    max_new_tokens=req.max_tokens or 400,
                    temperature=req.temperature or 0.7,
                    do_sample=True
                )
                raw_json = _parse_llm_json(outputs[0]["generated_text"][-1]["content"])

                if "location_id" not in raw_json or not raw_json["location_id"]:
                    raw_json["location_id"] = req.location.location_id
                if "node_id" not in raw_json or not raw_json["node_id"]:
                    raw_json["node_id"] = req.node_id

                parsed = DialogueOutput(**raw_json)
                return parsed.model_dump()
            except Exception as e:
                print(f"⚠️ [Dialogue] 驗證失敗 ({attempt+1}/{max_retries}): {e}")

        print("❌ [Dialogue] 觸發 Fallback")
        return DialogueOutput(
            location_id=req.location.location_id, node_id=req.node_id,
            narration={"opening_hook": "周圍的空氣似乎凝結了。", "scene_description": "場景中暫時沒有變化。", "historical_note": None},
            npc_dialogue=[{"npc_id": getattr(npc, 'npc_id', 'npc_01'), "line": "不好意思，我剛剛恍神了，你能再說一次嗎？", "emotion": "neutral", "handoff_to": None}],
            player_choices=[{"choice_id": "retry", "text": "再試一次"}]
        ).model_dump()

    elif node_type == "overnight_transition":
        req = OvernightTransitionInput(**payload_dict)
        sys_prompt = f"""你是一個實境遊戲編劇。請嚴格輸出JSON。
今日總結：{", ".join([s.summary_text for s in req.day_summary])}
住宿地點：{req.accommodation.name} - {req.accommodation.description}
請回傳 recap, accommodation_scene, next_day_hint。
"""
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": "請根據今日行程與住宿產出過夜轉場 JSON："}]

        for attempt in range(max_retries):
            try:
                outputs = llm(msgs, max_new_tokens=req.max_tokens or 400, temperature=req.temperature or 0.7, do_sample=True)
                raw_json = _parse_llm_json(outputs[0]["generated_text"][-1]["content"])

                if "day_index" not in raw_json:
                    raw_json["day_index"] = req.day_index

                parsed = OvernightTransitionOutput(**raw_json)
                return parsed.model_dump()
            except Exception as e:
                print(f"⚠️ [Overnight] 驗證失敗 ({attempt+1}/{max_retries}): {e}")

        print("❌ [Overnight] 觸發 Fallback")
        return OvernightTransitionOutput(
            day_index=req.day_index,
            recap="充實的一天結束了，回顧今日的旅程，收穫滿滿。",
            accommodation_scene=f"你回到了{req.accommodation.name}，舒適的環境讓你放鬆下來。",
            next_day_hint="好好休息吧，明天還有全新的挑戰等著你。"
        ).model_dump()

    elif node_type == "narration":
        req = NarrationInput(**payload_dict)
        sys_prompt = "你是一個旁白修飾引擎。請微調並輸出JSON。"
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"旁白原文：{req.script_text}\n請輸出JSON："}]

        for attempt in range(max_retries):
            try:
                outputs = llm(msgs, max_new_tokens=req.max_tokens or 400, temperature=req.temperature or 0.7, do_sample=True)
                raw_json = _parse_llm_json(outputs[0]["generated_text"][-1]["content"])

                if "node_id" not in raw_json:
                    raw_json["node_id"] = req.node_id
                if "day_index" not in raw_json:
                    raw_json["day_index"] = req.day_index

                parsed = NarrationOutputNode(**raw_json)
                return parsed.model_dump()
            except Exception as e:
                print(f"⚠️ [Narration] 驗證失敗 ({attempt+1}/{max_retries}): {e}")

        print("❌ [Narration] 觸發 Fallback")
        return NarrationOutputNode(
            day_index=req.day_index, node_id=req.node_id, narration_text=req.script_text
        ).model_dump()

    else:
        raise ValueError(f"未知的 node_type: {node_type}")


# ============================================================
# 3. 後台工具：結合 Neo4j 開放資料的實境劇本與NPC對話自動生成 (支援多參數與動態動線)
# ============================================================
def generate_script_blueprint(
    city_name: str,
    town_name: str,
    locations: list,
    traveler_count: int = 1,
    preferences: list = None,
    transportation: list = None,
    is_night: bool = False
) -> dict:
    preferences = preferences or []
    transportation = transportation or []

    llm = _get_llm()
    node_count = len(locations)

    locations_text = ""
    for idx, loc in enumerate(locations):
        loc_uuid = loc.get('uuid') or loc.get('id', '')
        node_type_ch = {"Attraction": "景點", "Restaurant": "特色餐廳", "Hotel": "在地旅宿"}.get(loc['type'], "據點")
        tags_str = f"、特色標籤：{', '.join(loc['tags'])}" if loc.get('tags') else ""
        cuisines_str = f"、推薦料理：{', '.join(loc['cuisines'])}" if loc.get('cuisines') else ""
        ticket_str = f"、門票資訊：{loc['ticket_info']}" if loc.get('ticket_info') else ""

        locations_text += f"- 第 {idx+1} 站 [{node_type_ch}]: {loc['name']} (UUID: {loc_uuid})\n"
        locations_text += f"  介紹摘要: {(loc.get('description') or '')[:200]}...\n"
        locations_text += f"  真實資料: 地址 {loc.get('address', '')}{tags_str}{cuisines_str}{ticket_str}\n\n"

    mode_title = "夜間尋密與過夜線" if is_night else "白天主線探索"

    npc_pool = [
        {"name": "薯光", "role": "充滿朝氣的新生代引導者，善於解開謎題、點燃線索"},
        {"name": "珍奶奶", "role": "掌管地方數十年的記憶與失落古老配方的守密人"},
        {"name": "阿達力", "role": "機靈靈通的在地走透透達人，熟悉大街小巷與美食情報"},
        {"name": "墨先生", "role": "博學嚴謹的文史工作者，擅長解讀古地圖與歷史檔案"},
        {"name": "霓霓", "role": "對美感與光影極度敏銳的街頭藝術家，專門引導光影觀察與夜遊探索"},
        {"name": "阿吉伯", "role": "外冷內熱的傳統工藝老師傅，重視手作與傳承"}
    ]
    selected_npc = random.choice(npc_pool)

    pref_str = "、".join(preferences) if preferences else "綜合體驗"
    trans_str = "、".join(transportation) if transportation else "大眾運輸與步行"

    system_prompt = f"""你是一位頂尖的中文歷史懸疑小說家與劇作家。
【絕對最高指令】：
1. 本次生成的所有欄位內容（包含標題、前言、大綱、角色介紹、任務說明、NPC開場白、過關對話等）**必須 100% 使用純正的繁體中文 (zh-TW) 撰寫**。
2. **絕對嚴禁出現任何英文字母、英文單字或英文句子**（唯獨 node 裡的 spot_uuid 必須原封不動填入對應站點提供的 UUID 字串）。

【任務目標】：請根據以下目標地區與真實地點，自行發揮創意構思一個引人入勝的解謎冒險主題，並為這 **{node_count} 個地點** 創作一篇情節豐富、細節飽滿的長篇解謎劇本（{mode_title}）。
- 目標地區：{city_name} {town_name}
- 旅程人數：{traveler_count} 人
- 旅客偏好：{pref_str}
- 交通方式：{trans_str}

【指定擔任主角的 NPC】：
- 姓名：{selected_npc['name']}
- 身分：{selected_npc['role']}

【指定站點清單 (共 {node_count} 站)】：
{locations_text}

請嚴格輸出純 JSON 格式 (不要包含任何 Markdown 標記)：
{{
  "title": "富有詩意、懸疑感且精采吸睛的劇本名稱（純繁體中文）",
  "preface": "大約100字以內的前言...",
  "synopsis": "四到五句、極具史詩感與懸疑氛圍的故事大綱...",
  "is_night_mode": {str(is_night).lower()},
  "npc": {{
    "name": "{selected_npc['name']}",
    "role": "{selected_npc['role']}",
    "intro": "詳細的角色小傳..."
  }},
  "nodes": [
    {{
      "node_order": 1,
      "spot_uuid": "請填入該站點對應的 UUID 字串",
      "place_name": "對應清單中的站點真實名稱",
      "location_codename": "一個非地點名稱卻能詩意描述該地點的代名稱",
      "node_title": "充滿意境的關卡標題（純繁體中文）",
      "task_type": "3. 採訪蒐證型",
      "task_description": "融合地點歷史與解謎線索的深度任務說明...",
      "dialogues": {{
        "opening": "長篇且充滿懸疑感的開場對白...",
        "success": "過關時的讚許..."
      }}
    }}
  ]
}}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "請嚴格遵守繁體中文指令，並將每個站點的 UUID 正確對應填入 JSON 中："}
    ]

    max_retries = 2
    # 依站點數動態給足 token，避免 JSON 被截斷後一直重試
    max_tokens = max(2500, 800 * node_count)

    for attempt in range(max_retries):
        response = ""
        try:
            outputs = llm(messages, max_new_tokens=max_tokens, temperature=0.7, do_sample=True)
            response = outputs[0]["generated_text"][-1]["content"]
            start, end = response.find("{"), response.rfind("}") + 1
            if start != -1 and end != 0:
                result_json = json.loads(response[start:end])

                # 🛡️ 確保 nodes 中的 spot_uuid 一定正確對應到 locations 資料
                if "nodes" in result_json:
                    for i, node in enumerate(result_json["nodes"]):
                        if i < len(locations):
                            node["spot_uuid"] = locations[i].get('uuid') or locations[i].get('id', '')

                return result_json
            raise ValueError("回應中找不到完整 JSON")
        except Exception as e:
            print(f"⚠️ 劇本藍圖生成失敗，重試中 ({attempt+1}/{max_retries})... 錯誤: {e}")
            print(f"   原始輸出後 300 字: ...{response[-300:]}")

    # Fallback 結構也補上 spot_uuid
    fallback_nodes = []
    for idx, loc in enumerate(locations):
        fallback_nodes.append({
            "node_order": idx + 1,
            "spot_uuid": loc.get('uuid') or loc.get('id', ''),
            "place_name": loc['name'],
            "location_codename": "時光站點",
            "node_title": f"探索 {loc['name']}",
            "task_type": "1. 探索型",
            "task_description": "探索當地的歷史足跡與文化痕跡。",
            "dialogues": {"opening": "歡迎來到這裡，開始你的冒險吧！", "success": "做得好！"}
        })

    return {
        "story_id": "fallback_01",
        "title": f"{town_name}時光迷局",
        "preface": "一段未知的歷史迷局，正等待著你的雙腳去解開...",
        "synopsis": "探索在地文化的精采旅程。",
        "is_night_mode": is_night,
        "nodes": fallback_nodes
    }


# ============================================================
# 4. Agent：意圖解析、行事曆工具、單站關卡企劃
# ============================================================
def parse_user_intent_agent(user_text: str, emotion: str) -> dict:
    """
    Step 2: 意圖解析 (Agent Brain)
    將模糊的語音輸入與情緒，轉換為精準的 Neo4j 查詢參數。
    """
    llm = _get_llm()

    sys_prompt = f"""你是一個精準的旅遊意圖解析 Agent。
【玩家情緒】：{emotion}
【玩家語音】：{user_text}

請判斷玩家當前的身心狀態，並輸出對應的實體搜尋條件。
若玩家感到「疲憊/炎熱」，請加上「室內」、「冷氣」、「靜態」等標籤。
若玩家感到「興奮/開心」，可以加上「戶外」、「探險」等標籤。

請嚴格輸出純 JSON：
{{
  "search_tags": ["室內", "冷氣", "咖啡廳"], 
  "activity_level": "low", 
  "radius_km": 1,
  "reasoning": "玩家表示走得很累，情緒疲憊，需要低強度的室內休息空間。"
}}"""

    messages = [{"role": "system", "content": sys_prompt}]

    try:
        outputs = llm(messages, max_new_tokens=300, temperature=0.1, do_sample=True)
        response = outputs[0]["generated_text"][-1]["content"]
        return _parse_llm_json(response)
    except Exception as e:
        print(f"⚠️ 意圖解析失敗: {e}")
        return {"search_tags": [], "activity_level": "medium", "radius_km": 3, "reasoning": "解析失敗，採用預設值"}


def tool_call_create_calendar_event(spot_name: str, address: str, script_intro: str) -> str:
    """
    Step 4: Tool Calling (排入行事曆)
    Agent 主動呼叫外部工具，將行程封裝成行事曆事件連結。
    """
    # 預設排入未來一小時的行程
    start_time = datetime.now(timezone.utc) + timedelta(hours=1)
    end_time = start_time + timedelta(hours=2)

    # Google Calendar URL 格式轉換 (YYYYMMDDTHHMMSSZ)
    fmt = "%Y%m%dT%H%M%SZ"
    dates = f"{start_time.strftime(fmt)}/{end_time.strftime(fmt)}"

    base_url = "https://calendar.google.com/calendar/render?action=TEMPLATE"
    params = {
        "text": f"🗺️ [Play Taiwan 解謎] 前往 {spot_name}",
        "dates": dates,
        "details": f"大富翁解謎劇本任務：\n\n{script_intro}\n\n準備好你的解謎包，我們不見不散！",
        "location": address,
    }

    return base_url + "&" + urllib.parse.urlencode(params)


def generate_single_spot_puzzle(spot_name: str, spot_desc: str, emotion: str) -> dict:
    """
    對應 Phase 4: 企劃與行動 (動態劇本與行前注意事項)
    """
    llm = _get_llm()
    sys_prompt = f"""你是一個精準的實境遊戲企劃 Agent。
【絕對語言指令】：
所有的文字內容（包含標題、對白、任務與提示）必須 100% 使用純正的台灣繁體中文 (zh-TW) 撰寫，絕對嚴禁出現任何英文單字或簡體字！

請根據以下地點與玩家情緒，快速設計一個專屬的「大富翁解謎劇本關卡」。
【地點】：{spot_name}
【簡介】：{(spot_desc or '')[:200]}
【玩家情緒】：{emotion}

請嚴格輸出純 JSON（請確保逗號不遺漏，並且內容全是繁體中文）：
{{
  "theme_title": "充滿懸疑感的關卡標題 (純繁體中文)",
  "npc_dialogue": "NPC(例如霓霓或薯光)的生動開場白與謎面提示 (純繁體中文)",
  "task_mission": "玩家需要完成的具體任務或尋寶目標 (純繁體中文)",
  "preparation_tips": ["行前注意事項1 (如：帶雨傘或防蚊液)", "行前注意事項2 (如：補充水分)"]
}}"""
    messages = [{"role": "system", "content": sys_prompt}]

    try:
        outputs = llm(messages, max_new_tokens=800, temperature=0.7, do_sample=True)
        response = outputs[0]["generated_text"][-1]["content"]

        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != 0:
            json_str = response[start:end]
            json_str = re.sub(r'\}\s*\{', '}, {', json_str)
            return json.loads(json_str)
    except Exception as e:
        print(f"⚠️ 關卡企劃生成失敗: {e}")

    # Fallback 防呆
    return {
        "theme_title": f"{spot_name} 的失落記憶",
        "npc_dialogue": f"歡迎來到 {spot_name}，看來你累了，先在這裡稍微休息一下，準備好我們再迎接挑戰！",
        "task_mission": "尋找隱藏在角落的秘密圖騰",
        "preparation_tips": ["帶著一顆放鬆的心", "喝口水補充水分"]
    }


def parse_script_request_agent(user_input: str) -> dict:
    """
    將使用者輸入的自然語言（文字或語音轉文字），解析成劇本藍圖所需的參數。
    """
    llm = _get_llm()
    sys_prompt = f"""你是一個精準的劇本參數解析 Agent。
請從使用者的自然語言輸入中，提取或推導出以下欄位：
1. city_name (字串，例如：臺南市、臺中市，預設為「臺南市」)
2. town_name (字串，例如：中西區、安平區，預設為「中西區」)
3. traveler_count (整數，人數，預設為 2)
4. preferences (字串陣列，旅客偏好，例如：["解謎深入", "文學建築"])
5. transportation (字串陣列，交通方式，例如：["步行", "公車"])
6. node_count (整數，站點數量，預設為 4)
7. is_night (布林值，是否為夜間模式，預設為 false)

使用者輸入內容：{user_input}

請嚴格輸出純 JSON 物件（不要包含任何 Markdown 標記）：
{{
  "city_name": "臺南市",
  "town_name": "中西區",
  "traveler_count": 2,
  "preferences": ["解謎深入", "文學建築"],
  "transportation": ["步行", "公車"],
  "node_count": 4,
  "is_night": false
}}"""

    messages = [{"role": "system", "content": sys_prompt}]
    try:
        outputs = llm(messages, max_new_tokens=400, temperature=0.1, do_sample=True)
        response_text = outputs[0]["generated_text"][-1]["content"]
        return _parse_llm_json(response_text)
    except Exception as e:
        print(f"⚠️ 劇本自然語言意圖解析失敗: {e}")
        # 回傳預設值
        return {
            "city_name": "臺南市",
            "town_name": "中西區",
            "traveler_count": 2,
            "preferences": ["解謎深入", "文學建築"],
            "transportation": ["步行", "公車"],
            "node_count": 4,
            "is_night": False
        }
import os
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
    NarrationInput, NarrationOutputNode,
    StoryTaskResponse
)
from transformers import pipeline
from core.config import TEMPLATES

# 啟動時印出版本與檔案路徑，確認伺服器跑的是這一份
SERVICE_VERSION = "batch-v7 (重試用完改用保底內容，不再回 500)"
print(f"📄 llm_service {SERVICE_VERSION} 載入自 {__file__}")

# ============================================================
# LLM 載入與執行緒安全
# ============================================================
_llm_pipeline = None
_llm_wrapper = None
_load_lock = threading.Lock()   # 避免多個請求同時載入模型
_gen_lock = threading.Lock()    # 同一時間只讓一個請求使用 GPU 生成

# 是否把輸出自動轉成台灣繁體 (需 pip install opencc-python-reimplemented)，設 LLM_OPENCC=0 可關閉
LLM_OPENCC = os.environ.get("LLM_OPENCC", "1") != "0"
_opencc_cc = None


def _to_tw(text: str) -> str:
    """簡體 → 台灣繁體（含用詞轉換）。沒裝 opencc 就原樣返回"""
    global _opencc_cc
    if not LLM_OPENCC:
        return text
    try:
        if _opencc_cc is None:
            import opencc
            _opencc_cc = opencc.OpenCC("s2twp")
        return _opencc_cc.convert(text)
    except Exception:
        return text


class _LockedLLM:
    """包一層鎖與計時，所有 llm(...) 呼叫點不用改"""

    def __init__(self, pipe):
        self._pipe = pipe

    def __call__(self, *args, **kwargs):
        with _gen_lock:
            t0 = time.time()
            out = self._pipe(*args, **kwargs)
            dt = time.time() - t0
            try:
                text = out[0]["generated_text"][-1]["content"]
                n = len(self._pipe.tokenizer(text).input_ids)
                print(f"⏱️ LLM 生成耗時 {dt:.1f}s，約 {n} tokens，{n / max(dt, 0.001):.1f} tok/s")
                # 統一做繁體轉換，所有函式都受惠
                out[0]["generated_text"][-1]["content"] = _to_tw(text)
            except Exception:
                print(f"⏱️ LLM 生成耗時 {dt:.1f}s")
            return out


# ============================================================
# 可用環境變數調整：預設使用 unsloth/Llama-3.2-3B-Instruct
# (要換模型可設 LLM_MODEL，例如 taide/Llama-3.1-TAIDE-LX-8B-Chat)
# ============================================================
LLM_MODEL = os.environ.get("LLM_MODEL", "unsloth/Llama-3.2-3B-Instruct")

# 3B 用無損 bfloat16 只佔約 6.4GB VRAM，不需要量化
LLM_QUANT = os.environ.get("LLM_QUANT", "none").lower()

# 批次生成時一次並行幾筆 (3B 很省 VRAM，ComfyUI 沒在算圖時可調到 24)
LLM_BATCH_SIZE = int(os.environ.get("LLM_BATCH_SIZE", "12"))


def _should_quantize() -> bool:
    if LLM_QUANT == "4bit":
        return True
    if LLM_QUANT == "none":
        return False
    if not torch.cuda.is_available():
        return False
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    return total_gb < 20


def _build_pipeline(use_4bit: bool):
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        return pipeline(
            "text-generation",
            model=LLM_MODEL,
            model_kwargs={"quantization_config": bnb},
            device_map="auto",
        )
    # 舊版 transformers 不認得 dtype 時，改回 model_kwargs={"torch_dtype": torch.bfloat16}
    return pipeline(
        "text-generation",
        model=LLM_MODEL,
        dtype=torch.bfloat16,
        device_map="auto",
    )


def _report_device(pipe):
    """印出模型實際放在哪裡，方便發現被丟到 CPU 的情況"""
    print(f"✅ LLM 已載入，裝置: {pipe.model.device}")
    dm = getattr(pipe.model, "hf_device_map", None)
    if dm:
        places = set(str(v) for v in dm.values())
        print(f"   device_map 使用位置: {places}")
        if any(p in ("cpu", "disk") for p in places):
            print("   ⚠️ 警告：部分模型層被放在 CPU/磁碟，生成會非常慢！"
                  "請關閉其他佔用 VRAM 的程式，或設定 LLM_QUANT=4bit。")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {props.name}，總 VRAM {props.total_memory / 1e9:.1f}GB，"
              f"本程序已用 {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    else:
        print("   ⚠️ 警告：torch 偵測不到 CUDA，模型會跑在 CPU 上！")


def _get_llm():
    global _llm_pipeline, _llm_wrapper
    if _llm_wrapper is None:
        with _load_lock:
            if _llm_wrapper is None:
                use_4bit = _should_quantize()
                print(f"🚀 正在載入 LLM 大腦 ({LLM_MODEL}，{'4bit 量化' if use_4bit else 'bfloat16'})...")
                try:
                    _llm_pipeline = _build_pipeline(use_4bit)
                except Exception as e:
                    if use_4bit:
                        print(f"⚠️ 4bit 載入失敗 ({e})，改用 bfloat16 重試 (需先安裝 bitsandbytes)")
                        _llm_pipeline = _build_pipeline(False)
                    else:
                        raise
                _report_device(_llm_pipeline)
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
            {"role": "user", "content": f"玩家剛才說：「{req.player_input}」。請根據上述規範產生對應 JSON 回應。"}
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
# NPC 角色池 (藍圖生成與批次劇本共用)
# ============================================================
NPC_POOL = [
    {"name": "薯光", "role": "充滿朝氣的新生代引導者，善於解開謎題、點燃線索"},
    {"name": "珍奶奶", "role": "掌管地方數十年的記憶與失落古老配方的守密人"},
    {"name": "阿達力", "role": "機靈靈通的在地走透透達人，熟悉大街小巷與美食情報"},
    {"name": "墨先生", "role": "博學嚴謹的文史工作者，擅長解讀古地圖與歷史檔案"},
    {"name": "霓霓", "role": "對美感與光影極度敏銳的街頭藝術家，專門引導光影觀察與夜遊探索"},
    {"name": "阿吉伯", "role": "外冷內熱的傳統工藝老師傅，重視手作與傳承"},
]


def _pick_npcs(n: int) -> list:
    """幫 n 份劇本各挑一個 NPC，數量夠就不重複"""
    if n <= len(NPC_POOL):
        return random.sample(NPC_POOL, n)
    return random.choices(NPC_POOL, k=n)


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

    selected_npc = random.choice(NPC_POOL)

    pref_str = "、".join(preferences) if preferences else "綜合體驗"
    trans_str = "、".join(transportation) if transportation else "大眾運輸與步行"

    system_prompt = f"""你是一位頂尖的中文歷史懸疑小說家與劇作家。
【絕對最高指令】：
1. 本次生成的所有欄位內容（包含標題、前言、大綱、角色介紹、任務說明、NPC開場白、過關對話等）**必須 100% 使用純正的繁體中文 (zh-TW) 撰寫**。
2. **絕對嚴禁出現任何英文字母、英文單字或英文句子**（唯獨 node 裡的 spot_uuid 必須原封不動填入對應站點提供的 UUID 字串）。
3. 請嚴格控制字數：preface 約 80 字、synopsis 約 100 字、npc.intro 約 60 字；每一站的 task_description 約 80 字、opening 約 60 字、success 約 30 字，不要超出。

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
        "opening": "充滿懸疑感的開場對白（約 60 字）...",
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
    # 已在 prompt 限制字數，依站點數給足 token 即可 (4 站約 2400)
    max_tokens = max(2000, 600 * node_count)

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


# ============================================================
# 5. 批次劇本任務生成 (Story & Task Batch Generation)
#    策略：story 層（含 NPC 小傳）一次批次 → 所有節點（含 NPC 台詞）攤平成大批次並行生成
#          → Python 端依規格書組裝欄位，失敗只重試該筆
#    prompt 附參考範例 (few-shot)，但一定挑「地點不在這次請求裡」的範例，避免 3B 直接照抄
#    重試時把上一次被退回的原因告訴模型；最後一輪放寬品質檢查，只保留格式與外文檢查
#    重試全部用完仍失敗的劇本或節點，改用 _fallback_story / _fallback_node 保底，不回 500
#    回傳在規格書之外多帶兩個欄位：stories[].npc、nodes[].dialogues
# ============================================================

# 預期可獲得的勳章 (規格書範例為「初心探員」)；請求若有帶 story_badge 就沿用請求的
_DEFAULT_BADGE = ["初心探員"]

# 每一筆最多生成幾輪 (每輪只重跑失敗的那幾筆，所以多一輪成本不高)
_MAX_ROUNDS = 4

# ---------- 欄位名稱正規化 ----------
_SEAT_KEYS = ("seat_no", "party_member", "clue_target", "seat", "player", "player_no", "member")
_TEXT_KEYS = ("clue_text", "clue", "clue_context", "text", "content", "hint", "description")


def _pick(d: dict, keys, default=None):
    for k in keys:
        if d.get(k) not in (None, ""):
            return d[k]
    return default


def _normalize_clue(c, index: int) -> dict:
    """統一 LLM 亂飄的 key；座位號缺失時用順序補上，文字缺失時取最長的字串值"""
    if isinstance(c, str):
        return {"seat_no": index, "clue_text": c}
    if not isinstance(c, dict):
        return {"seat_no": index, "clue_text": str(c)}

    try:
        seat = int(_pick(c, _SEAT_KEYS, default=index))
    except (TypeError, ValueError):
        seat = index

    text = _pick(c, _TEXT_KEYS)
    if not text:
        candidates = [
            v for k, v in c.items()
            if k not in _SEAT_KEYS and isinstance(v, str) and v.strip()
        ]
        text = max(candidates, key=len) if candidates else ""
    return {"seat_no": seat, "clue_text": str(text)}


def _normalize_story_result(result: dict) -> dict:
    for story in result.get("stories", []):
        for node in story.get("nodes", []):
            for task in node.get("tasks", []):
                clues = task.get("task_clue")
                if isinstance(clues, (dict, str)):
                    clues = [clues]
                if clues:
                    task["task_clue"] = [
                        _normalize_clue(c, i) for i, c in enumerate(clues, start=1)
                    ]
    return result


_BATCH_SYS = """你是頂尖的實境解謎遊戲編劇，擅長把在地文史寫成有畫面、有懸念的故事。
1. 必須 100% 使用繁體中文 (zh-TW)，不可夾雜任何英文單字與簡體字。
2. 旁白欄位（story_prologue 與 sn_ 開頭的欄位）不可寫 NPC 的台詞；sn_ 開頭的欄位一律用第二人稱「你/你們」。
3. 只有 npc_say_ 開頭的欄位是 NPC 親口對玩家說的話：口語自然、符合 NPC 的身分個性，直接寫內容，不要加「某某：」或引號。
4. npc_intro 用第三人稱介紹 NPC。
5. 句子要短，寫出這個地點獨有的材質、顏色、聲音或氣味，禁止「發現了許多秘密」「成功解開了謎題」這類空話。
6. 任務要是玩家在現場真的做得到的事（抬頭觀察、拍照、詢問店家、點一樣東西）。
7. 只回傳合法的純 JSON，字串裡要引用文字請用「」，不可用半形雙引號；不要 Markdown 標記，不要多餘說明。"""


# ---------- 參考範例 (few-shot)：每類準備兩個不同地點，挑不在這次請求裡的那個 ----------
# _place 只用來避開同地點，不會送進 prompt
_STORY_EXAMPLES = [
    {   # 規格書範例
        "_place": "臺中州廳",
        "story_title": "州廳鐘聲裡的家書",
        "story_prologue": "一封寄不出去的家書，被夾在老州廳的公文之間，收信人的名字早已模糊。",
        "story_synopsis": "循著百年建築的線索，替家書找到回家的路。",
        "npc_intro": "在老街顧了四十年舊書攤，最會辨認泛黃信紙上的字跡。",
    },
    {
        "_place": "赤崁樓",
        "story_title": "古堡帳本的最後一頁",
        "story_prologue": "一本荷蘭時代的帳本，最後一頁被人撕走，只留下一枚褪色的紅印。",
        "story_synopsis": "沿著府城老街，找回帳本缺少的那一頁。",
        "npc_intro": "在廟口擺了三十年字畫攤，看過的老文件比誰都多。",
    },
]

_NODE_EXAMPLES = {
    # 選擇題型 (6/7)
    "choice": [
        {
            "_place": "臺中州廳",
            "sn_title": "紅磚廊下的公文",
            "location_codename": "紅磚迴廊",
            "sn_opening_text": "你走進長長的迴廊，陽光從拱窗灑下，家書的第一個線索就藏在這棟建築裡。",
            "sn_success_text": "你認出了屋頂的樣式，家書上模糊的地址清楚了一些。",
            "npc_say_opening": "這棟樓的屋頂可是有來頭的，抬頭仔細看看吧。",
            "npc_say_success": "沒錯！信封上的郵戳，指向下一條老街。",
            "task_describe": "抬頭看看這棟建築的屋頂，它是哪一種樣式？",
            "task_hint": "這種屋頂源自法國，斜面很陡，上面還開了小窗。",
            "options": ["馬薩式屋頂", "燕尾脊", "歇山頂", "攢尖頂"],
        },
        {
            "_place": "赤崁樓",
            "sn_title": "城牆磚縫裡的帳本",
            "location_codename": "贔屭石碑",
            "sn_opening_text": "你走過紅磚老牆，九座石碑一字排開，馱碑的贔屭低著頭，像在守著什麼。",
            "sn_success_text": "你說出了建城者的來歷，帳本上的第一個記號亮了起來。",
            "npc_say_opening": "這座樓比你想的還老，先猜猜最早是誰蓋的？",
            "npc_say_success": "答對了！帳本下一頁畫的是一口古井。",
            "task_describe": "赤崁樓的前身普羅民遮城，最早是哪一國人建造的？",
            "task_hint": "他們在大航海時代來到臺灣，也在安平蓋了熱蘭遮城。",
            "options": ["荷蘭人", "西班牙人", "葡萄牙人", "英國人"],
        },
    ],
    # 協作解謎型 (5)
    "coop": [
        {
            "_place": "刑務所演武場",
            "sn_title": "木地板上的半個字",
            "location_codename": "木造道場",
            "sn_opening_text": "你們踏上吱吱作響的木地板，牆上的匾額只剩兩個殘缺的字。",
            "sn_success_text": "兩人拼出了匾額上的字，當年比試的項目終於揭曉。",
            "npc_say_opening": "匾額上的字被拆開了，你們一人只看得到一半。",
            "npc_say_success": "就是這兩個字！當年的勝負，答案藏在市場裡。",
            "task_describe": "匾額上的兩個字被拆開了，兩人各自只看得到一個，互相描述才能拼出答案。",
            "task_hint": "這座道場以前是練什麼的？",
            "correct_answer": "劍道",
            "clues": ["你看到的字左邊是「僉」，右邊是一把刀。", "你看到的字是「首」字加上走路的「辶」。"],
        },
        {
            "_place": "鹿港天后宮",
            "sn_title": "香煙裡的兩個字",
            "location_codename": "香爐前庭",
            "sn_opening_text": "你們站在燻黑的香爐前，煙霧散開時，籤筒上刻的字只剩一半。",
            "sn_success_text": "兩人拼出了廟裡供奉的神明，籤詩上的下一站浮現出來。",
            "npc_say_opening": "籤筒上的字被燻得看不清，你們一人記一半吧。",
            "npc_say_success": "沒錯，就是祂！籤詩說下一站在老街轉角。",
            "task_describe": "籤筒上的兩個字被香煙燻掉一半，兩人各自只看得到一個字的線索，互相描述才能拼出答案。",
            "task_hint": "這座廟供奉的是海上的守護神。",
            "correct_answer": "媽祖",
            "clues": ["你看到的字左邊是「女」，右邊是「馬」。", "你看到的字左邊是「示」字旁，右邊是「且」。"],
        },
    ],
    # 創意攝影型 (3)，也當 GPS 定位、跨關集結等其他打卡型的範例
    "photo": [
        {
            "_place": "國立臺灣美術館",
            "sn_title": "空白畫框前的合照",
            "location_codename": "空白畫框",
            "sn_opening_text": "展場牆上掛著一個空畫框，旁邊的說明牌寫著：請自行補上。",
            "sn_success_text": "你們的照片填滿了畫框，畫留下的腳印指向一條老巷。",
            "npc_say_opening": "畫框裡的畫跑掉了，得找人頂替一下才行。",
            "npc_say_success": "拍得真好！畫的腳印往老巷去了，跟上吧。",
            "task_describe": "找一個你們覺得最適合當畫作的角落，兩人一起入鏡拍一張照片。",
            "task_hint": "戶外的草地和雕塑也算展場的一部分。",
        },
        {
            "_place": "九份老街",
            "sn_title": "燈籠下的雙人照",
            "location_codename": "燈籠石階",
            "sn_opening_text": "你們踩上濕滑的石階，一整排紅燈籠在霧裡亮起，遠處看得見海。",
            "sn_success_text": "照片裡的燈籠多了一盞，底下掛著一塊寫了字的木牌。",
            "npc_say_opening": "這裡的燈籠會藏東西，拍張照就知道了。",
            "npc_say_success": "看到沒？多出來的那盞，就是下一站的方向。",
            "task_describe": "找一段看得到紅燈籠和遠方海景的石階，兩人一起入鏡拍一張照片。",
            "task_hint": "往豎崎路走，石階最陡的地方視野最好。",
        },
    ],
    # 地方美食型 (4)
    "food": [
        {
            "_place": "審計新村",
            "sn_title": "老宿舍的點心時間",
            "location_codename": "老宿舍",
            "sn_opening_text": "你跟著腳印來到一間老宿舍前，空氣裡有甜甜的味道。",
            "sn_success_text": "你們嚐到了畫最愛的點心，終於在攤位後面找到它。",
            "npc_say_opening": "聞到了嗎？那幅畫肯定躲在這附近偷吃。",
            "npc_say_success": "找到了！它嘴邊還沾著糖粉呢。",
            "task_describe": "在這裡找一樣在地小點心，拍下它並說出最特別的味道。",
            "task_hint": "問問看攤主，哪一樣是他們的招牌。",
        },
        {
            "_place": "基隆廟口夜市",
            "sn_title": "廟口的第一碗",
            "location_codename": "黃燈攤位",
            "sn_opening_text": "你們擠進一整排黃色燈籠下，鍋裡的湯冒著白煙，攤位編號一路排到廟門口。",
            "sn_success_text": "碗底印著一個號碼，正好對上下一站的攤位。",
            "npc_say_opening": "別急著吃，先找到開最久的那一攤。",
            "npc_say_success": "就是這碗！記住碗底的號碼。",
            "task_describe": "找一攤招牌上寫著創立年份的老攤，點一樣東西並拍下來。",
            "task_hint": "攤位越靠近奠濟宮，通常歷史越久。",
        },
    ],
    # e 人訪談型 (8)
    "talk": [
        {
            "_place": "範例旅館",
            "sn_title": "旅店櫃檯的最後一站",
            "location_codename": "旅店櫃檯",
            "sn_opening_text": "夜深了，你推開旅店的門，櫃檯老闆說這條街上的老故事他都聽過。",
            "sn_success_text": "老闆認出了家書上的姓氏，家書終於有了歸處。",
            "npc_say_opening": "最後一站了，問問老闆吧，他什麼都知道。",
            "npc_say_success": "原來收信人就住在這條街，信終於送到了。",
            "task_describe": "向旅店老闆打聽，這附近最老的一間店開了多久。",
            "task_hint": "找到櫃檯那位總是笑臉迎人的人。",
        },
        {
            "_place": "迪化街",
            "sn_title": "南北貨行的老秤",
            "location_codename": "老秤櫃檯",
            "sn_opening_text": "你們走進堆滿乾貨的老店，紅豆和香菇的味道撲鼻，櫃檯上擺著一支銅秤。",
            "sn_success_text": "老闆說出了秤的年紀，秤桿上的刻痕原來是一串日期。",
            "npc_say_opening": "那支秤比我還老，去問問老闆它的故事。",
            "npc_say_success": "刻痕是日期！下一站就照這個日子去找。",
            "task_describe": "向店家打聽櫃檯上那支老秤用了多少年，以前都秤些什麼。",
            "task_hint": "等老闆忙完一位客人再開口，他會很樂意多聊幾句。",
        },
    ],
}

_TYPE_TO_EXAMPLE = {6: "choice", 7: "choice", 5: "coop", 3: "photo", 4: "food", 8: "talk"}

# 這些敘事欄位不可照抄範例：整句相同，或連續 10 個字相同都算
_COPY_CHECK_KEYS = ("story_title", "story_prologue", "story_synopsis", "npc_intro",
                    "sn_title", "sn_opening_text", "sn_success_text",
                    "npc_say_opening", "npc_say_success")
_COPY_MIN = 10
_EXAMPLE_TEXTS = [
    v for ex in (*_STORY_EXAMPLES, *(e for pool in _NODE_EXAMPLES.values() for e in pool))
    for k, v in ex.items() if k in _COPY_CHECK_KEYS
]


# ---------- 小模型 (3B) 防呆 ----------
# 小模型常把範本裡的說明文字原封不動抄回來，出現這些字樣就視為失敗、只重試該筆
_PLACEHOLDER_MARKS = (
    "劇本標題", "懸念序幕", "交代起因與目標", "角色小傳",
    "有文學感與懸疑感", "角落或物件取名", "第二人稱旁白", "親口說的話",
    "實際做到的任務", "具體明確的提示", "錯誤選項", "線索片段", "簡短詞語",
)
_WORD_COUNT_MARK = re.compile(r"約\s*\d+\s*字")

# 小模型常在選項前自己加「A.」「(B)」，之後順序會被打亂，所以先去掉
_OPTION_PREFIX = re.compile(r"^\s*[\(（]?[A-Da-d][\)）\.．、:：]\s*")

# 3B 常夾雜英文或越南文 (如 lobby、waiting for you、ngon)；兩個字母以上的拉丁字串視為外文
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")

# 旁白裡出現「某某說『……』」代表 NPC 台詞寫錯欄位了（台詞應寫在 npc_say_ 欄位）
_NPC_SPEECH = re.compile(r"說\s*[：:]?\s*[「『“\"]")
_NARRATION_KEYS = ("story_prologue", "sn_opening_text", "sn_success_text")


def _text(v) -> str:
    """轉成去頭尾空白的字串；None 回傳空字串，並修掉 3B 常見的「你們們」疊字"""
    if v is None:
        return ""
    return re.sub(r"([你我他她])們們", r"\1們", str(v).strip())


def _norm_place(s) -> str:
    return _text(s).replace("台", "臺").replace(" ", "")


def _has_foreign(value, allow: str = "") -> bool:
    """有不在 allow (景點名稱、題型名稱) 裡的英文字串就算外文，例如允許「GPS 區域定位型」的 GPS"""
    return any(w not in allow for w in _LATIN_WORD.findall(str(value or "")))


def _is_placeholder(value) -> bool:
    s = _text(value)
    return bool(_WORD_COUNT_MARK.search(s)) or any(m in s for m in _PLACEHOLDER_MARKS)


def _is_copied(value) -> bool:
    s = _text(value)
    for ex in _EXAMPLE_TEXTS:
        if s == ex:
            return True
        if len(s) >= _COPY_MIN and any(ex[i:i + _COPY_MIN] in s
                                       for i in range(len(ex) - _COPY_MIN + 1)):
            return True
    return False


def _pick_example(pool: list, avoid_places: list) -> dict:
    """挑一個地點不在這次請求裡的範例 (去掉 _ 開頭的內部欄位)；全部撞到就用最後一個"""
    avoid = [_norm_place(a) for a in avoid_places if a]
    chosen = pool[-1]
    for ex in pool:
        p = _norm_place(ex["_place"])
        if not any(p in a or a in p for a in avoid):
            chosen = ex
            break
    return {k: v for k, v in chosen.items() if not k.startswith("_")}


def _clean_option(o) -> str:
    return _OPTION_PREFIX.sub("", _text(o)).strip()


def _clean_line(v, npc_name: str) -> str:
    """NPC 台詞去掉模型自己加的「薯光：」前綴與外層引號，說話者由 npc.name 提供"""
    s = _text(v)
    s = re.sub(rf"^{re.escape(npc_name)}\s*(?:說)?\s*[：:]\s*", "", s)
    if len(s) >= 2 and s[0] in "「『“\"" and s[-1] in "」』”\"":
        s = s[1:-1].strip()
    return s


_CJK_RANGE = "\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"


def _scrub_text(s: str, allow: str = "") -> str:
    """刪掉夾雜的英文單字 (景點名稱、題型名稱裡的除外)，並收掉刪除後留下的多餘空白"""
    new = re.sub(r"[A-Za-z]{2,}", lambda m: m.group(0) if m.group(0) in allow else "", s)
    if new != s:
        new = re.sub(rf"(?<=[{_CJK_RANGE}])[ \t]+|[ \t]+(?=[{_CJK_RANGE}])", "", new)
    return new.strip()


def _scrub_obj(obj, allow: str = ""):
    if isinstance(obj, str):
        return _scrub_text(obj, allow)
    if isinstance(obj, list):
        return [_scrub_obj(x, allow) for x in obj]
    if isinstance(obj, dict):
        return {k: _scrub_obj(v, allow) for k, v in obj.items()}
    return obj


def _strip_place(codename: str, p_name: str) -> str:
    """把場景代號裡的地名拿掉 (臺/台 視為同字)，例如「台中州廳迴廊」→「迴廊」"""
    pat = "".join("[臺台]" if ch in "臺台" else re.escape(ch) for ch in _text(p_name).replace(" ", ""))
    if not pat:
        return codename
    return re.sub(pat, "", codename).strip(" ，、:：-－")


def _check_text(key: str, value, allow: str = "", strict: bool = True) -> None:
    """
    欄位檢查。硬性 (每輪都檢查)：不可空白、不可抄範本說明、不可夾雜外文
    品質 (strict 才檢查，最後一輪放寬)：不可照抄範例、旁白不可有 NPC 台詞
    """
    if not _text(value):
        raise ValueError(f"欄位 {key} 為空")
    if _is_placeholder(value):
        raise ValueError(f"欄位 {key} 抄了範本說明文字")
    if _has_foreign(value, allow):
        raise ValueError(f"欄位 {key} 夾雜外文: {_LATIN_WORD.findall(str(value))}")
    if not strict:
        return
    if key in _COPY_CHECK_KEYS and _is_copied(value):
        raise ValueError(f"欄位 {key} 照抄了參考範例")
    if key in _NARRATION_KEYS and _NPC_SPEECH.search(_text(value)):
        raise ValueError(f"欄位 {key} 是旁白卻寫了 NPC 台詞")


def _retry_note(err: str) -> str:
    """重試時把上一次被退回的原因告訴模型，比單純重抽有效"""
    if not err:
        return ""
    if any(w in err for w in ("Expecting", "delimiter", "JSON", "char ")):
        err = "JSON 格式錯誤，字串裡不可出現半形雙引號，引用文字請改用「」"
    elif "夾雜外文" in err:
        err = err + "。整段文字必須全是繁體中文，任何英文單字都不行，想不到中文就換個說法"
    return f"\n\n⚠️ 你上一次的輸出被退回，原因：{err}。這次請務必修正。"


_NODE_KEYS = ("sn_title", "location_codename", "sn_opening_text", "sn_success_text",
              "npc_say_opening", "npc_say_success", "task_describe", "task_hint",
              "options", "correct_answer", "clues")


def _lenient_extract(text: str, keys) -> dict:
    """
    JSON 壞掉時的最後手段：依已知欄位名稱把值一段一段切出來。
    能處理 3B 常見的錯：字串裡有沒跳脫的半形雙引號、欄位之間漏逗號
    """
    end = text.rfind("}")
    if end != -1:
        text = text[:end + 1]
    marks = []
    for k in keys:
        m = re.search(rf'"{re.escape(k)}"\s*:', text)
        if m:
            marks.append((m.start(), m.end(), k))
    marks.sort()
    if not marks:
        raise ValueError("找不到任何已知欄位")
    out = {}
    for n, (_, en, k) in enumerate(marks):
        stop = marks[n + 1][0] if n + 1 < len(marks) else len(text)
        chunk = text[en:stop].strip().rstrip("}, \n\r\t")
        if chunk.startswith("["):
            inner = chunk[1:chunk.rfind("]")] if "]" in chunk else chunk[1:]
            parts = re.split(r'"\s*,\s*"', inner.strip())
            out[k] = [p.strip().strip('"').strip() for p in parts if p.strip().strip('"').strip()]
        elif chunk.startswith('"'):
            val = re.sub(r'"\s*$', "", chunk[1:])
            out[k] = val.replace('"', "").strip()
        else:
            out[k] = None if chunk in ("null", "") else chunk
    return out


def _extract_json(text: str, keys=None) -> dict:
    """
    取出第一個完整的 JSON 物件，依序嘗試：標準解析 (允許字串內換行) →
    json-repair (有裝才用) → 依欄位名稱容錯切割 (需傳入 keys)
    """
    s = text.find("{")
    if s == -1:
        raise ValueError("找不到 JSON")
    body = re.sub(r",\s*([}\]])", r"\1", text[s:])
    try:
        obj, _ = json.JSONDecoder(strict=False).raw_decode(body)
    except json.JSONDecodeError:
        obj = None
        try:
            from json_repair import repair_json
            obj = repair_json(body, return_objects=True)
        except Exception:
            obj = None
        if not isinstance(obj, dict) or not obj:
            if not keys:
                raise
            obj = _lenient_extract(body, keys)
    if not isinstance(obj, dict) or not obj:
        raise ValueError("JSON 不是物件")
    return obj


def _batch_generate(prompts: list, max_new_tokens: int) -> list:
    """一次把多個 prompt 丟給 GPU 並行生成，回傳與 prompts 等長的字串清單"""
    llm = _get_llm()
    pipe = llm._pipe
    tok = pipe.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"  # decoder-only 模型批次生成必須左側 padding

    msgs = [
        [{"role": "system", "content": _BATCH_SYS},
         {"role": "user", "content": p}]
        for p in prompts
    ]
    with _gen_lock:
        t0 = time.time()
        outs = pipe(
            msgs,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_p=0.9,          # 砍掉低機率的尾巴，可明顯減少夾英文
            do_sample=True,
            batch_size=len(msgs),
        )
        print(f"⏱️ 批次生成 {len(msgs)} 筆，耗時 {time.time() - t0:.1f}s")
    return [_to_tw(o[0]["generated_text"][-1]["content"]) for o in outs]


def _first_type(place: dict) -> tuple:
    tl = place.get("type_list") or [{}]
    return tl[0].get("type_id", 1), tl[0].get("type_name", "")


def _story_prompt(req_dict: dict, s: dict, npc: dict, s_tag: list, avoid: list) -> str:
    place_names = "、".join(p.get("p_name", "") for p in s.get("places", []))
    fields = {
        "story_title": "劇本標題",
        "story_prologue": "一句懸念序幕，約35字",
        "story_synopsis": "一句話交代起因與目標，約20字",
        "npc_intro": f"用第三人稱介紹{npc['name']}在本劇本中的身分與動機（角色小傳），約30字",
    }
    example = _pick_example(_STORY_EXAMPLES, avoid)
    return (
        f"地區：{req_dict.get('city_name', '')}{req_dict.get('district_name', '')}\n"
        f"使用者偏好：{'、'.join(s_tag)}\n"
        f"敘事語氣：{s.get('nt_name', '懸疑推理')}\n"
        f"引導 NPC：{npc['name']}（{npc['role']}）\n"
        f"依序經過：{place_names}\n"
        "參考範例（別的地方的劇本，只學它的寫法：短、具體、有懸念；內容與題材都不可沿用）：\n"
        f"{json.dumps(example, ensure_ascii=False)}\n"
        "請把下面 JSON 的說明文字替換成實際內容（標題要有詩意與畫面感，且不可與其他劇本雷同）：\n"
        f"{json.dumps(fields, ensure_ascii=False, indent=2)}"
    )


def _node_prompt(story_ctx: dict, place: dict, idx: int, total: int,
                 party_size: int, is_night: bool) -> str:
    task_type, type_name = _first_type(place)
    p_name = place.get("p_name", "該地點")
    npc = story_ctx["npc"]
    name = npc["name"]

    fields = {
        "sn_title": "有文學感與懸疑感的節點標題，不可和劇本標題相同",
        "location_codename": "用這裡看得到的一個角落或物件取名，2到5個字，不可包含地名",
        "sn_opening_text": "第二人稱旁白，描寫走進此地看到的具體細節，約35字",
        "sn_success_text": "第二人稱旁白，完成任務後劇情往前推進了什麼，約25字",
        "npc_say_opening": f"{name}在玩家抵達時親口說的話，約20字",
        "npc_say_success": f"{name}在玩家完成任務後親口說的話，約20字",
        "task_describe": "玩家在現場能實際做到的任務或題目",
        "task_hint": "具體明確的提示",
    }
    if task_type in (6, 7):
        fields["options"] = ["正確答案", "錯誤選項1", "錯誤選項2", "錯誤選項3"]
    elif task_type == 5:
        fields["correct_answer"] = "最終答案（簡短詞語）"
        fields["clues"] = [f"玩家{i + 1}看到的線索片段" for i in range(party_size)]

    pool = _NODE_EXAMPLES[_TYPE_TO_EXAMPLE.get(task_type, "photo")]
    example = _pick_example(pool, story_ctx["avoid"])

    extra = []
    # NPC 出場節奏：第一站自我介紹、中間帶出下一站線索、最後一站收尾
    if total == 1:
        extra.append(f"這是唯一一站：npc_say_opening 是{name}第一次和玩家見面，要簡短自我介紹；"
                     f"npc_say_success 要為整段故事收尾。")
    elif idx == 1:
        extra.append(f"這是第一站：npc_say_opening 是{name}第一次和玩家見面，要簡短自我介紹並交代這趟旅程要找什麼。")
    elif idx == total:
        extra.append(f"這是最後一站：npc_say_success 要由{name}為整段故事收尾。")
    else:
        extra.append(f"npc_say_success 要由{name}帶出前往下一站的線索。")
    if place.get("is_hotel") == 1:
        extra.append("此站是旅館，視為當天落腳處，劇情可作為收尾。")
    if place.get("is_hidden") == 1:
        extra.append("此站是隱藏關卡，劇情要寫成完成前一站才會發現的感覺。")
    if is_night:
        extra.append("劇情要有夜晚的氛圍。")
    if task_type in (6, 7):
        extra.append("題目要是問句，內容須符合該景點的真實資訊；options 陣列第一個必須是正確答案，共 4 個。")
    if task_type == 5:
        extra.append(f"clues 必須剛好 {party_size} 筆（範例是 2 人隊伍），"
                     f"且任何一位玩家單獨看自己的線索都無法推出答案。")

    return (
        f"劇本標題：{story_ctx['title']}\n"
        f"劇本大綱：{story_ctx['synopsis']}\n"
        f"敘事語氣：{story_ctx['tone']}\n"
        f"引導 NPC：{name}（{npc['role']}）\n"
        f"NPC 小傳：{story_ctx['npc_intro']}\n"
        f"目前是第 {idx}/{total} 站：【{p_name}】，任務類型：{type_name}\n"
        f"{' '.join(extra)}\n"
        f"參考範例（這是別的地方，只學它的寫法：句子短、畫面具體、任務在現場做得到；"
        f"句子與內容都不可沿用，必須寫【{p_name}】自己的特色）：\n"
        f"{json.dumps(example, ensure_ascii=False)}\n"
        f"請把下面 JSON 的說明文字替換成【{p_name}】的實際內容（欄位不可增減）：\n"
        f"{json.dumps(fields, ensure_ascii=False, indent=2)}"
    )


def _build_node(place: dict, idx: int, raw: dict, party_size: int,
                npc_name: str, story_title: str, strict: bool = True) -> dict:
    """依規格書組裝節點，欄位出現與否完全由 Python 控制；strict=False 時放寬品質檢查"""
    task_type, type_name = _first_type(place)
    p_name = place.get("p_name", "")
    allow = f"{p_name} {type_name}"

    # 場景代號含地名時先自動修掉 (剩 2 字以上才採用)，修不動才交給下面的檢查
    code = _text(raw.get("location_codename"))
    if p_name and code:
        fixed = _strip_place(code, p_name)
        if fixed != code and len(fixed) >= 2:
            raw = {**raw, "location_codename": fixed}

    for k in ("sn_title", "location_codename", "sn_opening_text",
              "sn_success_text", "task_describe", "task_hint"):
        _check_text(k, raw.get(k), allow, strict)

    if strict:
        if _text(raw["sn_title"]) == _text(story_title):
            raise ValueError("sn_title 和劇本標題一樣")
        if p_name and _norm_place(p_name) in _norm_place(raw["location_codename"]):
            raise ValueError("location_codename 直接用了地名")
        if task_type in (6, 7) and not re.search(r"[？?]", _text(raw["task_describe"])):
            raise ValueError("選擇題的 task_describe 不是問句")

    dialogues = {
        "opening": _clean_line(raw.get("npc_say_opening"), npc_name),
        "success": _clean_line(raw.get("npc_say_success"), npc_name),
    }
    _check_text("npc_say_opening", dialogues["opening"], allow, strict)
    _check_text("npc_say_success", dialogues["success"], allow, strict)

    task = {
        "task_type": task_type,
        "task_describe": _text(raw["task_describe"]),
        "task_hint": _text(raw["task_hint"]),
        "correct_answer": None,
    }

    if task_type in (6, 7):
        opts = raw.get("options")
        if not isinstance(opts, list) or len(opts) != 4:
            raise ValueError("選項數量不是 4 個")
        opts = [_clean_option(o) for o in opts]
        if any(not o or _is_placeholder(o) or "正確答案" in o for o in opts):
            raise ValueError("選項有空值或抄了範本說明文字")
        if any(_has_foreign(o, allow) for o in opts):
            raise ValueError("選項夾雜外文")
        if len(set(opts)) != 4:
            raise ValueError("選項內容重複")
        keys = ["A", "B", "C", "D"]
        order = list(range(4))
        random.shuffle(order)  # 打亂正確答案位置，避免固定放 A
        task["task_option"] = [
            {"option_key": keys[n], "option_context": opts[i],
             "is_correct": 1 if i == 0 else 0}
            for n, i in enumerate(order)
        ]
    elif task_type == 5:
        clues = raw.get("clues")
        if not isinstance(clues, list) or len(clues) != party_size:
            raise ValueError(f"線索數量不是 {party_size} 筆")
        # 小模型偶爾會把線索寫成物件，借用 _normalize_clue 取出文字
        texts = [
            "" if c is None else _text(_normalize_clue(c, i + 1)["clue_text"])
            for i, c in enumerate(clues)
        ]
        if any(not t or _is_placeholder(t) or _has_foreign(t, allow) for t in texts):
            raise ValueError("線索有空值、抄了範本說明文字或夾雜外文")
        answer = _text(raw.get("correct_answer"))
        if not answer or _is_placeholder(answer) or _has_foreign(answer, allow):
            raise ValueError("協作解謎缺少有效的 correct_answer")
        if any(answer in t for t in texts):
            raise ValueError("有玩家的線索直接寫出了答案")
        task["correct_answer"] = answer
        task["task_clue"] = [
            {"seat_no": i + 1, "clue_text": t} for i, t in enumerate(texts)
        ]

    return {
        "place_id": place.get("place_id"),
        "sn_order": idx,
        "sn_title": _text(raw["sn_title"]),
        "location_codename": _text(raw["location_codename"]),
        "sn_opening_text": _text(raw["sn_opening_text"]),
        "sn_success_text": _text(raw["sn_success_text"]),
        "dialogues": dialogues,
        "tasks": [task],
    }


# ---------- 保底：重試用完仍失敗時，用 Python 組出格式正確的簡單內容 ----------
_FALLBACK_TASK = {
    1: ("抵達【{p}】並完成定位打卡。", "走到這個地點的範圍內就會完成打卡。"),
    2: ("在【{p}】和隊友會合，確認大家都到齊後一起完成集結。", "先在入口附近找到隊友。"),
    3: ("在【{p}】找一個你最喜歡的角落，拍一張照片留念。", "試著把這裡最有特色的東西拍進去。"),
    4: ("在【{p}】附近找一樣在地的特色小吃，拍下來並說說它的味道。", "問問店家，哪一樣是招牌。"),
    8: ("向【{p}】的在地人打聽這裡最有名的故事，記下重點。", "挑對方不忙的時候開口，會聊得更久。"),
}


def _fallback_story(req_dict: dict, s: dict, npc: dict) -> dict:
    area = req_dict.get("district_name") or req_dict.get("city_name", "")
    return {
        "story_title": f"{area}尋蹤・{s.get('nt_name', '探索')}篇",  # 帶語氣名稱，避免三份標題重複
        "story_prologue": f"{npc['name']}留下一張地圖，上面標著幾個地點，等著你們逐一拜訪。",
        "story_synopsis": f"跟著{npc['name']}走訪{area}，完成每一站的任務。",
        "npc_intro": f"{npc['role']}。",
    }


def _fallback_node(place: dict, idx: int, total: int, party_size: int,
                   npc: dict, all_names: list) -> dict:
    task_type, _ = _first_type(place)
    p = place.get("p_name", "該地點")
    task = {"task_type": task_type, "correct_answer": None}

    if task_type in (6, 7):
        # 題目固定問「這是哪裡」，干擾選項用這次請求的其他景點，內容一定正確
        opts = [p]
        for n in list(dict.fromkeys(all_names)) + ["老街口", "市區公園", "火車站前廣場"]:
            if n and n not in opts and len(opts) < 4:
                opts.append(n)
        correct = opts[0]
        random.shuffle(opts)
        task["task_describe"] = "你現在所在的地點叫什麼名字？"
        task["task_hint"] = "看看入口的招牌或說明牌。"
        task["task_option"] = [
            {"option_key": k, "option_context": o, "is_correct": 1 if o == correct else 0}
            for k, o in zip("ABCD", opts)
        ]
    elif task_type == 5:
        # 把地名拆給每位玩家，單看自己的那一份推不出答案
        chars = list(p)
        task["task_describe"] = "每位玩家各自看得到景點名稱的一部分，互相描述才能拼出完整名稱。"
        task["task_hint"] = "先把各自看到的字說出來，再討論順序。"
        task["correct_answer"] = p
        clues = []
        for i in range(party_size):
            part = chars[i::party_size]
            text = f"你看到的字是：{'、'.join(part)}。" if part else "你這次沒有字，請仔細聽隊友描述。"
            clues.append({"seat_no": i + 1, "clue_text": text})
        task["task_clue"] = clues
    else:
        desc, hint = _FALLBACK_TASK.get(task_type, _FALLBACK_TASK[3])
        task["task_describe"] = desc.format(p=p)
        task["task_hint"] = hint

    last = idx == total
    return {
        "place_id": place.get("place_id"),
        "sn_order": idx,
        "sn_title": f"{p}的線索",
        "location_codename": "入口一帶",
        "sn_opening_text": f"你來到【{p}】，四周安靜下來，下一個線索就在附近。",
        "sn_success_text": "你完成了這一站的任務，故事又往前推進了一步。",
        "dialogues": {
            "opening": "到了！先在這裡四處看看吧。",
            "success": "辛苦了，這趟旅程到這裡就告一段落。" if last else "做得好，我們繼續往下一站走。",
        },
        "tasks": [task],
    }


def _round_mode(attempt: int, n_todo: int, layer: str) -> tuple:
    """
    回傳 (strict, scrub)：
      strict：是否做品質檢查 (照抄範例、標題、代號、問句、旁白台詞)，最後一輪關閉
      scrub ：是否直接刪掉夾雜的英文字而不是退回重寫，最後兩輪開啟
    """
    strict = attempt < _MAX_ROUNDS - 1
    scrub = attempt >= _MAX_ROUNDS - 2
    if n_todo and not strict:
        print(f"ℹ️ {layer}最後一輪：放寬照抄範例、標題、代號等品質檢查，只保留格式檢查")
    elif n_todo and scrub:
        print(f"ℹ️ {layer}第 {attempt + 1} 輪起：夾雜的英文字改為自動刪除，不再退回重寫")
    return strict, scrub


def generate_batch_story_tasks(req_dict: dict) -> dict:
    party_size = req_dict.get("party_size", 2)
    is_night = req_dict.get("is_night_mode") == 1
    s_tag = req_dict.get("s_tag", [])
    stories_req = req_dict.get("stories", [])
    total_t0 = time.time()

    # 這次請求出現的所有景點，挑參考範例時要避開
    avoid = [p.get("p_name", "") for s in stories_req for p in s.get("places", [])]

    # 每份劇本一個 NPC (從 NPC_POOL 挑，3 份劇本不重複)；請求若自帶 npc (需有 name) 就優先採用
    picked = _pick_npcs(len(stories_req))
    npcs = []
    for i, s in enumerate(stories_req):
        n = s.get("npc")
        if isinstance(n, dict) and n.get("name"):
            npcs.append({"name": n["name"], "role": n.get("role", "在地引導者")})
        else:
            npcs.append(picked[i])

    # ---- 第一步：所有劇本的 story 層 (含 NPC 小傳)，一次批次生成 ----
    story_prompts = [_story_prompt(req_dict, s, npcs[si], s_tag, avoid)
                     for si, s in enumerate(stories_req)]

    story_keys = ("story_title", "story_prologue", "story_synopsis", "npc_intro")
    story_infos = [None] * len(stories_req)
    story_errs = {}
    for attempt in range(_MAX_ROUNDS):
        todo = [i for i, v in enumerate(story_infos) if v is None]
        if not todo:
            break
        strict, scrub = _round_mode(attempt, len(todo), "故事層")
        outs = _batch_generate([story_prompts[i] + _retry_note(story_errs.get(i)) for i in todo], 500)
        for i, o in zip(todo, outs):
            try:
                d = _extract_json(o, story_keys)
                allow = " ".join(p.get("p_name", "") for p in stories_req[i].get("places", []))
                if scrub:
                    d = _scrub_obj(d, allow)
                for k in story_keys:
                    _check_text(k, d.get(k), allow, strict)
                story_infos[i] = d
            except Exception as e:
                story_errs[i] = str(e)
                print(f"⚠️ 劇本 {i + 1} 故事層失敗（會重試）: {e}")
                print(f"   原始輸出開頭: {o[:120]!r}")
    n_fallback = 0
    for i, v in enumerate(story_infos):
        if v is None:
            print(f"❌ 劇本 {i + 1} 故事層重試用完，改用保底內容")
            story_infos[i] = _fallback_story(req_dict, stories_req[i], npcs[i])
            n_fallback += 1

    # ---- 第二步：所有節點 (含 NPC 台詞) 攤平成一個大批次，分段並行生成 ----
    jobs = []  # (story_idx, place_idx, place, prompt)
    for si, s in enumerate(stories_req):
        ctx = {
            "title": _text(story_infos[si]["story_title"]),
            "synopsis": _text(story_infos[si]["story_synopsis"]),
            "npc_intro": _text(story_infos[si]["npc_intro"]),
            "tone": s.get("nt_name", "懸疑推理"),
            "npc": npcs[si],
            "avoid": avoid,
        }
        places = s.get("places", [])
        for pi, p in enumerate(places):
            jobs.append((si, pi, p,
                         _node_prompt(ctx, p, pi + 1, len(places), party_size, is_night)))

    results = {}
    node_errs = {}
    for attempt in range(_MAX_ROUNDS):
        todo = [j for j in jobs if (j[0], j[1]) not in results]
        if not todo:
            break
        strict, scrub = _round_mode(attempt, len(todo), "節點層")
        for k in range(0, len(todo), LLM_BATCH_SIZE):
            chunk = todo[k:k + LLM_BATCH_SIZE]
            prompts = [c[3] + _retry_note(node_errs.get((c[0], c[1]))) for c in chunk]
            outs = _batch_generate(prompts, 800)
            for (si, pi, place, _), o in zip(chunk, outs):
                try:
                    raw = _extract_json(o, _NODE_KEYS)
                    if scrub:
                        raw = _scrub_obj(raw, f"{place.get('p_name', '')} {_first_type(place)[1]}")
                    results[(si, pi)] = _build_node(place, pi + 1, raw, party_size,
                                                    npcs[si]["name"],
                                                    story_infos[si]["story_title"],
                                                    strict)
                except Exception as e:
                    node_errs[(si, pi)] = str(e)
                    print(f"⚠️ 劇本{si + 1} 節點{pi + 1} 失敗（會重試）: {e}")
                    print(f"   原始輸出開頭: {o[:120]!r}")

    for si, pi, place, _ in [j for j in jobs if (j[0], j[1]) not in results]:
        print(f"❌ 劇本{si + 1} 節點{pi + 1} 重試用完，改用保底內容（最後原因：{node_errs.get((si, pi))}）")
        total = len(stories_req[si].get("places", []))
        results[(si, pi)] = _fallback_node(place, pi + 1, total, party_size, npcs[si], avoid)
        n_fallback += 1

    # ---- 第三步：組裝並驗證 ----
    stories = []
    for si, s in enumerate(stories_req):
        info = story_infos[si]
        badge = s.get("story_badge") or req_dict.get("story_badge") or _DEFAULT_BADGE
        stories.append({
            "story_no": s.get("story_no"),
            "story": {
                "story_title": _text(info["story_title"]),
                "story_prologue": _text(info["story_prologue"]),
                "story_synopsis": _text(info["story_synopsis"]),
                "story_badge": list(badge),
            },
            "npc": {
                "name": npcs[si]["name"],
                "role": npcs[si]["role"],
                "intro": _text(info["npc_intro"]),
            },
            "nodes": [results[(si, pi)] for pi in range(len(s.get("places", [])))],
        })

    note = f"，其中 {n_fallback} 處使用保底內容" if n_fallback else ""
    print(f"✅ 批次劇本任務完成，共 {len(jobs)} 個節點{note}，總耗時 {time.time() - total_t0:.1f}s")
    # exclude_unset：沒設定的 Optional 欄位 (如開放型的 task_option、task_clue) 不輸出成 null
    dumped = StoryTaskResponse.model_validate({"stories": stories}).model_dump(exclude_unset=True)

    # schemas.py 沒加新欄位時，pydantic 會默默丟掉，這裡提醒一下
    first = (dumped.get("stories") or [{}])[0]
    if "npc" not in first or "dialogues" not in ((first.get("nodes") or [{}])[0]):
        print("⚠️ api/schemas.py 尚未加入 npc / dialogues 欄位，這兩個欄位不會出現在回應裡")
    return dumped
import os
import uuid
import shutil
import json
import zipfile
import edge_tts
import urllib.parse
import time
import threading

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, HTTPException, Request, Body
from fastapi.responses import FileResponse, JSONResponse

from core.config import (
    IMAGES_DIR,
    AUDIO_DIR,
    BGM_DIR,
    OUTPUT_DIR
)

from services.asr_service import transcribe_audio
from services.llm_service import generate_vlog_content_with_template, parse_user_intent_agent, tool_call_create_calendar_event, generate_single_spot_puzzle, parse_script_request_agent
from services.tts_service import generate_tts
from services.moviepy_service import process_vlog_task
from services.ser_service import analyze_emotion
from services.postcard_service import (
    generate_postcard_text,
    generate_ai_postcard_image
)
from services.llm_service import generate_story_node, generate_script_blueprint
from services.neo4j_rag_service import search_neo4j_rag, execute_readonly_cypher, fetch_locations_for_script, fetch_spot_complete_info, find_spots_by_intent

router = APIRouter()

@router.post("/admin/upload_dump")
async def upload_dump_file(file: UploadFile = File(...)):
    try:
        target_path = OUTPUT_DIR.parent / file.filename
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "filename": file.filename, "message": "Dump 檔案上傳成功！"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

class CypherQueryRequest(BaseModel):
    query: str = Field(default="MATCH (a:Attraction) RETURN a.name LIMIT 5")
    parameters: Optional[Dict[str, Any]] = Field(default={})

@router.post("/neo4j/cypher")
async def execute_raw_cypher(req: CypherQueryRequest):
    try:
        records = execute_readonly_cypher(req.query, req.parameters)
        return {"status": "success", "count": len(records), "data": records}
    except Exception as e:
        return JSONResponse(status_code=400, content={"status": "error", "message": str(e)})

@router.post("/npc/speak")
async def npc_speak_api(
    text: str = Form("歡迎來到府城，請跟著我一起尋找失落的卷軸。"),
    voice: str = Form("zh-TW-HsiaoChenNeural")
):
    task_id = str(uuid.uuid4())[:8]
    output_filename = f"npc_{task_id}.mp3"
    output_path = OUTPUT_DIR / output_filename
    try:
        os.makedirs(str(OUTPUT_DIR), exist_ok=True)
        communicate = edge_tts.Communicate(text, voice, rate="+10%", pitch="+5Hz")
        await communicate.save(str(output_path))
        return {
            "status": "success",
            "task_id": task_id,
            "download_url": f"https://aiservice.angelalala.com/api/download/{output_filename}"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.post("/postcard/create_ai")
async def create_ai_postcard_api(
    user_image: UploadFile = File(...),
    spot_name: str = Form("臺南孔廟"),
    user_prompt: str = Form("復古水墨風")
):
    task_id = str(uuid.uuid4())[:8]
    try:
        os.makedirs(str(IMAGES_DIR), exist_ok=True)
        raw_img_path = str(IMAGES_DIR / f"postcard_raw_{task_id}.jpg")
        with open(raw_img_path, "wb") as buffer:
            shutil.copyfileobj(user_image.file, buffer)

        postcard_text = generate_postcard_text(spot_name, user_prompt)
        final_img_path = await generate_ai_postcard_image(raw_img_path, spot_name, user_prompt, task_id)
        return {
            "status": "success",
            "task_id": task_id,
            "postcard_introduction": postcard_text,
            "download_url": f"https://aiservice.angelalala.com/api/download/ai_postcard_{task_id}.png"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.get("/check_status/{task_id}")
async def check_status(task_id: str):
    try:
        for file_path in OUTPUT_DIR.glob(f"*{task_id}*.*"):
            if file_path.is_file() and not file_path.name.startswith("metadata_"):
                return {
                    "status": "ready",
                    "filename": file_path.name,
                    "download_url": f"https://aiservice.angelalala.com/api/download/{file_path.name}"
                }
        return {"status": "processing", "task_id": task_id}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@router.get("/download/{filename}")
async def download_file(filename: str):
    try:
        file_path = (OUTPUT_DIR / os.path.basename(filename)).resolve()
        if not file_path.exists():
            return JSONResponse(status_code=404, content={"message": "找不到檔案"})
        return FileResponse(path=file_path, filename=file_path.name)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

def process_uploads(task_id, image_zip, bgm_file):
    os.makedirs(str(IMAGES_DIR), exist_ok=True)
    os.makedirs(str(BGM_DIR), exist_ok=True)

    zip_temp_path = str(IMAGES_DIR / f"{task_id}_images.zip")
    extract_to = str(IMAGES_DIR / f"{task_id}_extracted")
    with open(zip_temp_path, "wb") as buffer:
        shutil.copyfileobj(image_zip.file, buffer)

    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_temp_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    full_image_paths = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                full_image_paths.append(os.path.join(root, file))

    bgm_path = None
    if bgm_file and bgm_file.filename:
        bgm_path = str(BGM_DIR / f"{task_id}_{bgm_file.filename}")
        with open(bgm_path, "wb") as buffer:
            shutil.copyfileobj(bgm_file.file, buffer)

    return full_image_paths, bgm_path

# ============================================================
# 模式一：遊客端 - 預覽與確認正式產片
# ============================================================
class SpotHistoryItem(BaseModel):
    spot_name: str
    location_codename: Optional[str] = ""
    visit_time: str = Field(default="2026-08-27 14:00")

class VisitorVlogPreviewRequest(BaseModel):
    spot_history: list[SpotHistoryItem]
    player_play_time: str = Field(default="2.5小時")
    game_tasks_completed: str = Field(default="完成解謎與尋寶任務")


@router.post("/visitor/vlog/preview")
async def visitor_vlog_preview_api(req: VisitorVlogPreviewRequest):
    try:
        enriched_spots = []
        rag_contexts = []

        # 🆕 使用 enumerate 來標示第幾站
        for idx, item in enumerate(req.spot_history):
            spot_info = fetch_spot_complete_info(item.spot_name)
            enriched_spots.append({
                "spot_name": spot_info["name"] or item.spot_name,
                "location_codename": item.location_codename,
                "visit_time": item.visit_time,
                "db_description": spot_info["description"]
            })
            
            # 🌟 核心修改：將 Neo4j 知識與「遊戲回憶」完美揉合
            if spot_info["description"]:
                tags_str = f" (特色: {', '.join(spot_info['tags'])})" if spot_info.get('tags') else ""
                memory_fragment = (
                    f"【第 {idx+1} 站 - {spot_info['name']}{tags_str}】\n"
                    f"📍 在地文化：{spot_info['description'][:150]}...\n"
                    f"🎮 玩家回憶：我在這裡與當地人(NPC)深度互動，成功破解了此處的隱藏任務，"
                    f"並在過關後獲得了一張屬於這裡的專屬【手繪紀念明信片】，真的非常有成就感！"
                )
                rag_contexts.append(memory_fragment)

        rag_context_str = "\n\n".join(rag_contexts) if rag_contexts else "資料庫無額外補充文史。"

        # 🌟 改變 emotion 為強烈的「回憶感」
        script_data = generate_vlog_content_with_template(
            raw_text=f"總遊玩時長: {req.player_play_time}, 達成目標: {req.game_tasks_completed}",
            emotion="感性、成就感、充滿回憶與溫度", 
            template="user_vlog",
            rag_context=rag_context_str,
            play_time=req.player_play_time,
            game_tasks=req.game_tasks_completed
        )

        return {
            "status": "success",
            "draft_preview": {
                "itinerary": enriched_spots,
                "suggested_script": script_data["tw_script"],
                "seo_keywords": script_data.get("seo_keywords", []),
                "promo_copy": script_data.get("promo_copy", "")
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# ============================================================
# 模式一：遊客端 - 確認後的正式影音合成端點 (改為非同步背景任務，防 502 逾時)
# ============================================================
@router.post("/visitor/vlog/create_final")
async def visitor_vlog_final_api(
    bg_tasks: BackgroundTasks,
    final_script: str = Form(..., description="經使用者確認或微調後的最終旁白"),
    image_zip: UploadFile = File(..., description="闖關過程中上傳的照片/影片壓縮檔"),
    bgm_file: UploadFile = File(..., description="背景音樂檔 (mp3)"),
    spot_meta_json: str = Form('[]', description="對應圖片與地點時間的中繼資料 JSON")
):
    task_id = str(uuid.uuid4())[:8]
    filename = f"vlog_visitor_{task_id}.mp4"

    try:
        # 1. 快速處理上傳檔案解壓縮與暫存 (確保秒速回應 API，不讓 Cloudflare 逾時)
        meta_mapping = json.loads(spot_meta_json)
        full_image_paths, bgm_path = process_uploads(task_id, image_zip, bgm_file)

        # 2. 依照前端傳來的 meta_mapping 順序對應圖片
        sorted_image_paths = []
        for file_info in meta_mapping:
            target_name = file_info.get("file_name", "")
            matched_path = next((p for p in full_image_paths if os.path.basename(p) == target_name), None)
            if matched_path:
                sorted_image_paths.append(matched_path)

        if not sorted_image_paths:
            sorted_image_paths = full_image_paths

        # 3. 預先產生 TTS 旁白與 VTT 動態字幕 
        tts_path, vtt_path = await generate_tts(final_script)

        # 4. 放入背景執行續 (BackgroundTasks) 執行耗時的 FFmpeg 影片合成，秒回 status: processing
        bg_tasks.add_task(
            process_vlog_task,
            task_id, sorted_image_paths, "cinematic lighting",
            tts_path, bgm_path, filename, "user_vlog",
            merchant_name="", subtitle_text=final_script,
            spot_metadata=meta_mapping,
            vtt_file=vtt_path  # 🆕 把字幕檔案路徑傳進去
        )

        return {
            "status": "processing",
            "task_id": task_id,
            "message": "影片正在背景加速合成中，請稍後透過 check_url 檢查狀態！",
            "check_url": f"https://aiservice.angelalala.com/api/check_status/{task_id}"
        }
    except Exception as e:
        print(f"❌ 遊客端最終產片失敗: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ============================================================
# 模式二：商家端
# ============================================================
@router.post("/merchant/vlog/preview")
async def merchant_vlog_preview_api(
    merchant_name: str = Form(...),
    promo_focus: str = Form(...)
):
    try:
        spot_info = fetch_spot_complete_info(merchant_name)
        if not spot_info["description"]:
            return JSONResponse(status_code=404, content={"status": "warning", "message": f"Neo4j 資料庫查無「{merchant_name}」資訊，不進行幻覺補全。"})

        rag_context = f"商家名稱: {spot_info['name']}\n地址: {spot_info['address']}\n官方介紹: {spot_info['description']}"
        script_data = generate_vlog_content_with_template(
            raw_text=f"推廣重點: {promo_focus}",
            emotion="平靜、放鬆、悠閒",
            template="merchant_promo",
            rag_context=rag_context,
            promo_info=promo_focus
        )
        return {"status": "success", "retrieved_data": spot_info, "draft_preview": script_data}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

# ============================================================
# 遊戲推演與劇本藍圖生成 API
# ============================================================
@router.post("/v1/generate")
async def generate_game_story(payload: dict = Body(...)):
    try:
        result_dict = generate_story_node(payload)
        return JSONResponse(status_code=200, content=result_dict)
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

class ScriptGenRequest(BaseModel):
    city_name: str = Field(default="臺南市")
    town_name: str = Field(default="中西區")
    traveler_count: int = Field(default=2)
    preferences: list[str] = Field(default=["解謎深入", "文學建築"])
    transportation: list[str] = Field(default=["步行", "公車"])
    node_count: int = Field(default=4)
    is_night: bool = Field(default=False)

@router.post("/admin/generate_script_blueprint")
async def api_generate_script_blueprint(req: ScriptGenRequest):
    try:
        locations = fetch_locations_for_script(town_name=req.town_name, limit=req.node_count, is_night=req.is_night)
        if not locations:
            return JSONResponse(status_code=400, content={"status": "error", "message": f"找不到 {req.town_name} 地點資料。"})

        script_blueprint = generate_script_blueprint(
            city_name=req.city_name,
            town_name=req.town_name,
            locations=locations,
            traveler_count=req.traveler_count,
            preferences=req.preferences,
            transportation=req.transportation,
            is_night=req.is_night
        )
        return {"status": "success", "data": script_blueprint}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

class AgentRequest(BaseModel):
    user_voice_transcript: str = Field(
        default="我走得好累又好熱，想找個有冷氣的地方休息一下",
        description="使用者語音轉文字的內容",
        json_schema_extra={"example": "我走得好累又好熱，想找個有冷氣的地方休息一下"}
    )
    emotion_label: str = Field(
        default="疲憊",
        description="使用者當下的情緒標籤",
        json_schema_extra={"example": "疲憊"}
    )
    user_lat: float = Field(
        default=24.150,
        description="使用者目前緯度",
        json_schema_extra={"example": 24.150}
    )
    user_lon: float = Field(
        default=120.683,
        description="使用者目前經度",
        json_schema_extra={"example": 120.683}
    )

@router.post("/agent/orchestrate")
async def execute_agent_workflow(req: AgentRequest):
    try:
        # Phase 1 & 2: 感知與意圖解析
        intent = parse_user_intent_agent(req.user_voice_transcript, req.emotion_label)
        
        # Phase 3: 知識圖譜檢索 (Graph RAG)
        spots = find_spots_by_intent(
            lat=req.user_lat, 
            lon=req.user_lon, 
            tags=intent.get("search_tags", []),
            max_distance_meters=intent.get("radius_km", 2) * 1000
        )
        
        if not spots:
            return {"message": "附近沒有符合您當下心情的地點。"}
            
        best_spot = spots[0]
        
        # Phase 4.1: 企劃與行動 - 動態劇本與行前提醒生成
        puzzle_data = generate_single_spot_puzzle(
            spot_name=best_spot['name'], 
            spot_desc=best_spot.get('description', ''), 
            emotion=req.emotion_label
        )
        
        # Phase 4.2: Tool Calling 1 - 封裝行事曆
        script_intro = f"{puzzle_data['theme_title']}\n\n💬 {puzzle_data['npc_dialogue']}\n🎯 任務：{puzzle_data['task_mission']}\n\n💡 貼心提醒：{', '.join(puzzle_data['preparation_tips'])}"
        
        calendar_link = tool_call_create_calendar_event(
            spot_name=best_spot['name'],
            address=best_spot.get('address', '臺中市'),
            script_intro=script_intro
        )
        
        # Phase 4.3: Tool Calling 2 - 社交邀請碼生成
        invite_code = str(uuid.uuid4())[:6].upper()
        social_share_link = f"https://aiservice.angelalala.com/invite?code={invite_code}&spot={urllib.parse.quote(best_spot['name'])}"
        
        # 回傳完美對應報告腳本的 JSON 結構
        return {
            "phase_1_perception": {
                "voice_input": req.user_voice_transcript,
                "emotion_detected": req.emotion_label
            },
            "phase_2_cognition": {
                "agent_thought_process": intent.get("reasoning", "分析中..."),
                "extracted_tags": intent.get("search_tags", [])
            },
            "phase_3_graph_rag": {
                "recommended_spot": best_spot
            },
            "phase_4_action_and_tools": {
                "script_blueprint": puzzle_data,
                "tool_1_calendar_sync": calendar_link,
                "tool_2_social_share": social_share_link
            },
            "phase_5_next_step": "等待前端確認後，背景觸發 TTS 語音朗讀與 Vlog 數位明信片生成"
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

class ScriptTextPromptRequest(BaseModel):
    user_prompt: str = Field(
        default="我們想要兩個人去臺南安平區來趟深度文學解謎之旅，大約走四個站點",
        description="使用者輸入的自然語言文字或語音轉文字內容",
        json_schema_extra={"example": "我們想要兩個人去臺南安平區來趟深度文學解謎之旅，大約走四個站點"}
    )

# ============================================================
# 劇本藍圖（自然語言版）：非同步任務，避免 Cloudflare 100 秒逾時 (524)
# ============================================================
BLUEPRINT_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 60 * 60  # 任務結果保留 1 小時


def _set_job(job_id: str, **fields):
    with _JOBS_LOCK:
        job = BLUEPRINT_JOBS.setdefault(job_id, {})
        job.update(fields)
        job["updated_at"] = time.time()


def _cleanup_old_jobs():
    now = time.time()
    with _JOBS_LOCK:
        expired = [j for j, v in BLUEPRINT_JOBS.items()
                   if now - v.get("updated_at", now) > JOB_TTL_SECONDS]
        for j in expired:
            BLUEPRINT_JOBS.pop(j, None)


def _run_blueprint_job(job_id: str, user_prompt: str):
    t0 = time.time()
    try:
        _set_job(job_id, status="running", stage="解析使用者需求")
        parsed = parse_script_request_agent(user_prompt)

        city_name = parsed.get("city_name") or "臺南市"
        town_name = parsed.get("town_name") or "中西區"
        traveler_count = parsed.get("traveler_count") or 2
        preferences = parsed.get("preferences") or ["解謎深入"]
        transportation = parsed.get("transportation") or ["步行"]
        if not isinstance(preferences, list):
            preferences = [str(preferences)]
        if not isinstance(transportation, list):
            transportation = [str(transportation)]
        try:
            node_count = max(1, min(int(parsed.get("node_count") or 4), 8))
        except (TypeError, ValueError):
            node_count = 4
        is_night = parsed.get("is_night") in (True, "true", "True")

        _set_job(job_id, stage="查詢 Neo4j 地點", parsed_intent=parsed)
        locations = fetch_locations_for_script(town_name=town_name, limit=node_count, is_night=is_night)
        if not locations:
            _set_job(job_id, status="error", error=f"找不到 {town_name} 地點資料。")
            return

        _set_job(job_id, stage="生成劇本藍圖")
        script_blueprint = generate_script_blueprint(
            city_name=city_name,
            town_name=town_name,
            locations=locations,
            traveler_count=traveler_count,
            preferences=preferences,
            transportation=transportation,
            is_night=is_night
        )

        _set_job(
            job_id,
            status="done",
            stage="完成",
            data=script_blueprint,
            elapsed_seconds=round(time.time() - t0, 1)
        )
        print(f"✅ [藍圖任務 {job_id}] 完成，耗時 {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"❌ [藍圖任務 {job_id}] 失敗: {e}")
        _set_job(job_id, status="error", error=str(e))


@router.post("/api/admin/generate_script_blueprint_by_text")
def api_generate_script_blueprint_by_text(
    req: ScriptTextPromptRequest,
    bg_tasks: BackgroundTasks,
    request: Request
):
    """立刻回傳 job_id，實際生成在背景執行。"""
    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex[:8]
    _set_job(job_id, status="queued", stage="排隊中", data=None, error=None)
    bg_tasks.add_task(_run_blueprint_job, job_id, req.user_prompt)
    return {
        "status": "processing",
        "job_id": job_id,
        "check_url": f"https://aiservice.angelalala.com{request.url.path.rstrip('/')}/{job_id}"
    }


@router.get("/api/admin/generate_script_blueprint_by_text/{job_id}")
def api_get_blueprint_job(job_id: str):
    with _JOBS_LOCK:
        job = BLUEPRINT_JOBS.get(job_id)
        job = dict(job) if job else None
    if not job:
        return JSONResponse(status_code=404, content={"status": "not_found", "job_id": job_id})
    job.pop("updated_at", None)
    return {"job_id": job_id, **job}
import logging
import asyncio
import json
import os
import shutil
import threading
import time
import urllib.parse
import uuid
import zipfile
from enum import Enum
from typing import Any, Dict, Optional

import edge_tts
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from api.story_generation_schemas import (
    StoryGenerationResult,
)

from services.story_workflow_service import (
    generate_story_workflow,
    internal_error_result,
)
from api.schemas import (
    SingleStoryTaskRequest,
    SingleStoryTaskResponse,
    StoryTaskRequest,
    StoryTaskResponse,
)
from core.config import (
    AUDIO_DIR,
    BGM_DIR,
    IMAGES_DIR,
    OUTPUT_DIR,
)
from services.asr_service import transcribe_audio
from services.bgm_service import select_bgm
from services.job_queue import job_queue
from services.llm_service import (
    analyze_vlog_music_style,
    generate_batch_story_tasks,
    generate_merchant_vlog_content,
    generate_script_blueprint,
    generate_single_spot_puzzle,
    generate_single_story_tasks,
    generate_story_node,
    generate_vlog_content_with_template,
    parse_script_request_agent,
    parse_user_intent_agent,
    tool_call_create_calendar_event,
)
from services.moviepy_service import process_vlog_task
from services.neo4j_rag_service import (
    execute_readonly_cypher,
    fetch_locations_for_script,
    fetch_spot_complete_info,
    find_spots_by_intent,
    search_neo4j_rag,
    recommend_locations_for_script,
)
from services.postcard_service import (
    generate_ai_postcard_image,
    generate_postcard_text,
)
from services.ser_service import analyze_emotion
from services.tts_service import generate_tts

from api.story_task_schemas import PublicStoryTaskRequest, PublicStoryTaskResponse
from services.story_tasks_service import generate_public_story_tasks


router = APIRouter()
logger = logging.getLogger(__name__)

@router.get(
    "/jobs/{task_id}",
    summary="查詢 AI 任務排隊與生成狀態",
)
async def get_ai_job(task_id: str):
    return job_queue.get(task_id)


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

class NPCChoice(str, Enum):
    zhen = "珍奶奶"
    adali = "阿達力"
    mo = "墨先生"
    aji = "阿吉伯"
    nini = "霓霓"
    shuguang = "薯光"
    wan = "王俊凱"

class VideoOrientation(str, Enum):
    portrait = "portrait"
    landscape = "landscape"


class BgmMood(str, Enum):
    auto = "auto"
    cheerful = "cheerful"
    relaxed = "relaxed"
    warm = "warm"
    nostalgic = "nostalgic"
    adventurous = "adventurous"
    neutral = "neutral"
    none = "none"

@router.post(
    "/npc/speak",
    summary="生成 NPC 語音（排隊模式）",
    status_code=202,
)
async def npc_speak_api(
    text: str = Form(
        "年輕人，這條街的故事我最熟了！",
        description="NPC 要說的台詞內容",
    ),
    npc_name: NPCChoice = Form(
        NPCChoice.zhen,
        description="請選擇發聲的 NPC 角色",
    ),
):
    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="台詞不可為空白",
        )

    if len(text) > 1000:
        raise HTTPException(
            status_code=400,
            detail="單次 NPC 台詞最多 1000 字",
        )

    task_id = uuid.uuid4().hex
    actual_npc_name = npc_name.value

    async def execute_job():
        job_queue.stage(task_id, "正在生成語音")

        audio_path, vtt_path = await generate_tts(
            text=text,
            npc_name=actual_npc_name,
        )

        output_filename = os.path.basename(audio_path)

        return {
            "filename": output_filename,
            "download_url": (
                "https://aiservice.angelalala.com"
                f"/api/download/{output_filename}"
            ),
        }

    return job_queue.submit(
        task_type="npc_tts",
        factory=execute_job,
        task_id=task_id,
    )


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
        # 僅接受任務 ID，避免被用來組出任意路徑
        if (
            not task_id
            or len(task_id) > 64
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in task_id)
        ):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "無效的 task_id"}
            )

        final_filename = f"vlog_visitor_{task_id}.mp4"
        final_path = OUTPUT_DIR / final_filename

        if final_path.is_file() and final_path.stat().st_size > 0:
            return {
                "status": "ready",
                "task_id": task_id,
                "filename": final_filename,
                "download_url": (
                    "https://aiservice.angelalala.com"
                    f"/api/download/{final_filename}"
                )
            }

        return {
            "status": "processing",
            "task_id": task_id
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

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
# 遊客端：自動配樂正式產片
# ============================================================
@router.post(
    "/visitor/vlog/create_final",
    summary="生成旅遊 Vlog（文字分析自動配樂、排隊模式）",
    status_code=202,
)
async def visitor_vlog_final_api(
    final_script: str = Form(
        ...,
        description="經使用者確認的最終旁白，不可空白，最多 5000 字。",
    ),
    image_zip: UploadFile = File(
        ...,
        description=(
            "照片 ZIP，支援 PNG、JPG、JPEG、WEBP。"
            "目前不支援將影片放進 ZIP 直接合成。"
        ),
    ),
    spot_meta_json: str = Form(
        ...,
        title="圖片與地點中繼資料 JSON",
        description="""每張圖片需有一筆資料，陣列順序就是播放順序。

必要欄位：
- file_name：ZIP 內實際圖片檔名，包含副檔名，區分大小寫。
- spot_name：影片左上角顯示的地點名稱。

選填欄位：
- location_codename：景點或任務代號。
- visit_time：造訪時間。
- motion_type：push_in、pull_out、pan_left、pan_right、still。

圖片在子資料夾時，file_name 仍只填檔名。
ZIP 不可有同名圖片，每張圖片必須恰好對應一筆資料。

範例：

```json
[
  {
    "file_name": "001.jpg",
    "spot_name": "環湖茶園",
    "location_codename": "茶香起點",
    "visit_time": "14:30",
    "motion_type": "push_in"
  },
  {
    "file_name": "002.jpg",
    "spot_name": "竹林步道",
    "location_codename": "竹影秘境",
    "visit_time": "15:20",
    "motion_type": "pan_right"
  },
  {
    "file_name": "003.jpg",
    "spot_name": "旅程紀念合照",
    "location_codename": "回憶終點",
    "visit_time": "16:10",
    "motion_type": "pull_out"
  }
]
```
""",
    ),
    user_memory_text: str = Form(
        "",
        description=(
            "選填，使用者原始旅遊感想，最多 3000 字。"
            "自動配樂會結合這段文字與最終旁白判斷風格。"
            "例如：和朋友一起完成任務，覺得很有成就感。"
            "未填時只分析 final_script。"
        ),
    ),
    bgm_mood: BgmMood = Form(
        BgmMood.auto,
        description=(
            "auto：分析文字自動選曲；"
            "cheerful：輕快；relaxed：悠閒；warm：溫馨；"
            "nostalgic：回憶；adventurous：探索；"
            "neutral：中性；none：不配樂。"
        ),
    ),
    video_orientation: VideoOrientation = Form(
        VideoOrientation.landscape,
        description=(
            "portrait：1080×1920；"
            "landscape：1920×1080。"
        ),
    ),
):
    final_script = final_script.strip()
    user_memory_text = user_memory_text.strip()

    if not final_script:
        raise HTTPException(
            status_code=400,
            detail="旁白不可為空白",
        )

    if len(final_script) > 5000:
        raise HTTPException(
            status_code=400,
            detail="旁白最多 5000 字",
        )

    if len(user_memory_text) > 3000:
        raise HTTPException(
            status_code=400,
            detail="原始旅遊感想最多 3000 字",
        )

    try:
        meta_mapping = json.loads(spot_meta_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "spot_meta_json 不是有效 JSON："
                f"第 {exc.lineno} 行、第 {exc.colno} 欄"
            ),
        ) from exc

    if (
        not isinstance(meta_mapping, list)
        or not meta_mapping
        or not all(isinstance(item, dict) for item in meta_mapping)
    ):
        raise HTTPException(
            status_code=400,
            detail="spot_meta_json 必須是非空的物件陣列",
        )

    valid_motion_types = {
        "push_in",
        "pull_out",
        "pan_left",
        "pan_right",
        "still",
    }

    normalized_metadata = []
    used_names = set()

    # 在寫入上傳檔案之前，先驗證 metadata 基本格式。
    for index, item in enumerate(meta_mapping, start=1):
        name = item.get("file_name")
        spot_name = item.get("spot_name")

        if not isinstance(name, str) or not name.strip():
            raise HTTPException(
                status_code=400,
                detail=f"第 {index} 筆缺少 file_name",
            )

        if not isinstance(spot_name, str) or not spot_name.strip():
            raise HTTPException(
                status_code=400,
                detail=f"第 {index} 筆缺少 spot_name",
            )

        name = name.strip()

        if name in used_names:
            raise HTTPException(
                status_code=400,
                detail=f"圖片重複指定：{name}",
            )

        normalized = dict(item)
        normalized["file_name"] = name
        normalized["spot_name"] = spot_name.strip()

        for key in ("location_codename", "visit_time"):
            value = item.get(key)

            if value is None:
                value = ""

            if not isinstance(value, str):
                raise HTTPException(
                    status_code=400,
                    detail=f"第 {index} 筆 {key} 必須是字串",
                )

            normalized[key] = value.strip()

        motion = item.get("motion_type")

        if motion is not None:
            if (
                not isinstance(motion, str)
                or motion not in valid_motion_types
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"第 {index} 筆 motion_type 無效",
                )
        else:
            normalized.pop("motion_type", None)

        used_names.add(name)
        normalized_metadata.append(normalized)

    task_id = uuid.uuid4().hex
    filename = f"vlog_visitor_{task_id}.mp4"
    orientation = video_orientation.value
    requested_bgm = bgm_mood.value

    # 使用既有照片上傳流程，不再要求使用者提供音樂。
    try:
        full_image_paths, _ = await asyncio.to_thread(
            process_uploads,
            task_id,
            image_zip,
            None,
        )
    except Exception as exc:
        print(f"[任務 {task_id}] 上傳處理失敗：{exc}")
        raise HTTPException(
            status_code=400,
            detail="照片 ZIP 處理失敗",
        ) from exc

    if not full_image_paths:
        raise HTTPException(
            status_code=400,
            detail="ZIP 內沒有支援的圖片",
        )

    image_by_name = {}

    for image_path in full_image_paths:
        name = os.path.basename(image_path)

        if name in image_by_name:
            raise HTTPException(
                status_code=400,
                detail=f"ZIP 有同名圖片，請先重新命名：{name}",
            )

        image_by_name[name] = image_path

    unknown_names = sorted(
        used_names - set(image_by_name)
    )
    missing_metadata = sorted(
        set(image_by_name) - used_names
    )

    print(
        f"[任務 {task_id}] ZIP 實際圖片："
        + json.dumps(
            sorted(image_by_name),
            ensure_ascii=False,
        )
    )

    print(
        f"[任務 {task_id}] metadata 指定圖片："
        + json.dumps(
            sorted(used_names),
            ensure_ascii=False,
        )
    )

    if unknown_names or missing_metadata:
        print(
            f"[任務 {task_id}] ❌ 配對失敗："
            f"找不到圖片={unknown_names!r}；"
            f"缺少地點資料={missing_metadata!r}"
        )

    if unknown_names or missing_metadata:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "圖片與地點資料無法完整配對",
                "metadata_file_not_found": unknown_names,
                "images_without_metadata": missing_metadata,
                "available_image_files": sorted(image_by_name),
            },
        )

    sorted_metadata = normalized_metadata
    sorted_image_paths = [
        image_by_name[item["file_name"]]
        for item in sorted_metadata
    ]

    for index, meta in enumerate(sorted_metadata, start=1):
        print(
            f"[任務 {task_id}] 🏷️ "
            f"{index}：{meta['file_name']}"
            f" → {meta['spot_name']}"
        )

    async def execute_job():
        bgm_path = None
        selected_track = None

        if requested_bgm == "none":
            music_analysis = {
                "mood": "none",
                "reason": "使用者選擇不加入背景音樂。",
                "analysis_source": "manual",
            }
        else:
            job_queue.stage(
                task_id,
                "正在分析配樂風格",
            )

            if requested_bgm == "auto":
                music_analysis = await asyncio.to_thread(
                    analyze_vlog_music_style,
                    user_type="visitor",
                    final_script=final_script,
                    original_text=user_memory_text,
                )
            else:
                music_analysis = {
                    "mood": requested_bgm,
                    "reason": "使用者指定配樂風格。",
                    "analysis_source": "manual",
                }

            job_queue.stage(
                task_id,
                "正在從本機曲庫選擇配樂",
            )

            selected_track = await asyncio.to_thread(
                select_bgm,
                mood=music_analysis["mood"],
                selection_key=task_id,
            )

            bgm_path = selected_track["path"]

            print(
                f"[任務 {task_id}] 🎵 "
                f"分析分類：{music_analysis['mood']}"
                f"｜實際分類：{selected_track['selected_mood']}"
                f"｜檔案：{selected_track['file_name']}"
                f"｜原因：{music_analysis['reason']}"
            )

        job_queue.stage(
            task_id,
            "正在生成 XTTS 旁白",
        )

        tts_path, vtt_path = await generate_tts(final_script)

        job_queue.stage(
            task_id,
            "正在合成影片",
        )

        await asyncio.to_thread(
            process_vlog_task,
            task_id,
            sorted_image_paths,
            "cinematic lighting",
            tts_path,
            bgm_path,
            filename,
            "user_vlog",
            merchant_name="",
            subtitle_text=final_script,
            spot_metadata=sorted_metadata,
            vtt_file=vtt_path,
            video_orientation=orientation,
        )

        final_path = OUTPUT_DIR / filename

        if not final_path.is_file() or final_path.stat().st_size == 0:
            raise RuntimeError("沒有產生有效的正式影片")

        # 對外不回傳伺服器本機絕對路徑。
        public_track = (
            {
                key: value
                for key, value in selected_track.items()
                if key != "path"
            }
            if selected_track
            else None
        )

        return {
            "filename": filename,
            "video_orientation": orientation,
            "music_analysis": music_analysis,
            "bgm": public_track,
            "download_url": (
                "https://aiservice.angelalala.com"
                f"/api/download/{filename}"
            ),
        }

    return job_queue.submit(
        task_type="vlog",
        factory=execute_job,
        task_id=task_id,
    )

# ============================================================
# 商家端：宣傳影片草稿預覽
# ============================================================
@router.post(
    "/merchant/vlog/preview",
    summary="商家宣傳影片預覽（商家資料、Neo4j、配樂建議）",
)
async def merchant_vlog_preview_api(
    merchant_name: str = Form(
        ...,
        description="商家名稱，不可空白。",
    ),
    promo_focus: str = Form(
        ...,
        description=(
            "本次宣傳重點，例如：介紹既有品茶體驗。"
            "想宣傳的項目仍須有商家介紹或資料庫內容支持。"
        ),
    ),
    merchant_id: str = Form(
        "",
        description=(
            "選填，Neo4j 商家 UID。"
            "有填時優先依 UID 查詢，不再用名稱備援配對。"
        ),
    ),
    merchant_info: str = Form(
        "",
        description=(
            "選填，商家自行提供的介紹、服務內容與特色。"
            "Neo4j 沒有可用介紹時，需提供此欄。"
            "最多 4000 字。"
        ),
    ),
    target_audience: str = Form(
        "",
        description="選填，目標客群，例如：喜歡慢旅行的旅客。",
    ),
    brand_style: str = Form(
        "",
        description="選填，品牌風格，例如：自然、溫暖、安靜。",
    ),
    promotion_goal: str = Form(
        "",
        description="選填，宣傳目的，例如：增加導覽詢問。",
    ),
    call_to_action: str = Form(
        "",
        description=(
            "選填，商家已確認的結尾引導。"
            "會直接附加到旁白及文案，不讓模型另編電話或網址。"
        ),
    ),
):
    fields = {
        "merchant_name": merchant_name.strip(),
        "promo_focus": promo_focus.strip(),
        "merchant_id": merchant_id.strip(),
        "merchant_info": merchant_info.strip(),
        "target_audience": target_audience.strip(),
        "brand_style": brand_style.strip(),
        "promotion_goal": promotion_goal.strip(),
        "call_to_action": call_to_action.strip(),
    }

    limits = {
        "merchant_name": 120,
        "promo_focus": 1000,
        "merchant_id": 200,
        "merchant_info": 4000,
        "target_audience": 300,
        "brand_style": 300,
        "promotion_goal": 500,
        "call_to_action": 500,
    }

    for key, value in fields.items():
        if len(value) > limits[key]:
            raise HTTPException(
                status_code=400,
                detail=f"{key} 最多 {limits[key]} 字",
            )

    if not fields["merchant_name"]:
        raise HTTPException(
            status_code=400,
            detail="商家名稱不可空白",
        )

    if not fields["promo_focus"]:
        raise HTTPException(
            status_code=400,
            detail="宣傳重點不可空白",
        )

    lookup_key = (
        fields["merchant_id"]
        or fields["merchant_name"]
    )

    warnings_list = []

    try:
        spot_info = await asyncio.to_thread(
            fetch_spot_complete_info,
            lookup_key,
        )
    except Exception as exc:
        print(f"⚠️ [商家預覽] Neo4j 查詢異常：{exc}")
        spot_info = {}
        warnings_list.append(
            "商家資料查詢失敗；若有商家提供資訊，將使用該資訊。"
        )

    if not isinstance(spot_info, dict):
        spot_info = {}

    # 依目前 helper 的回傳慣例：
    # 有非空 id，才視為取得資料庫紀錄。
    # 查不到時 helper 會回傳輸入名稱，所以不能只看 name。
    has_db_record = bool(spot_info.get("id"))

    db_info = (
        spot_info
        if has_db_record
        else {}
    )

    db_description = str(
        db_info.get("description") or ""
    ).strip()

    if not has_db_record:
        warnings_list.append(
            "未取得可確認的資料庫紀錄；"
            "這也可能是資料不足或查詢服務不可用。"
        )

    elif not db_description:
        warnings_list.append(
            "已取得資料庫紀錄，但沒有可用介紹。"
        )

    db_name = str(db_info.get("name") or "").strip()

    if db_name and db_name != fields["merchant_name"]:
        warnings_list.append(
            f"資料庫名稱為「{db_name}」，"
            f"輸入名稱為「{fields['merchant_name']}」，"
            "請確認是否為同一商家。"
        )

    if not db_description and not fields["merchant_info"]:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "沒有足夠的商家介紹可生成宣傳稿。"
                    "請補充 merchant_info，"
                    "不能只靠商家名稱與宣傳目標編造服務。"
                ),
                "warnings": warnings_list,
            },
        )

    source_records = []

    if has_db_record:
        source_records.append({
            "source": "neo4j",
            "id": str(db_info.get("id")),
            "name": db_name,
            "description_available": bool(db_description),
        })

    if fields["merchant_info"]:
        source_records.append({
            "source": "merchant_input",
            "fields": ["merchant_info"],
        })

    try:
        draft = await asyncio.to_thread(
            generate_merchant_vlog_content,
            merchant_name=fields["merchant_name"],
            promo_focus=fields["promo_focus"],
            merchant_info=fields["merchant_info"],
            db_info=db_info,
            target_audience=fields["target_audience"],
            brand_style=fields["brand_style"],
            promotion_goal=fields["promotion_goal"],
            call_to_action=fields["call_to_action"],
        )

        music_suggestion = await asyncio.to_thread(
            analyze_vlog_music_style,
            user_type="merchant",
            final_script=draft["tw_script"],
            original_text=(
                f"商家介紹：{fields['merchant_info']}\n"
                f"宣傳重點：{fields['promo_focus']}\n"
                f"目標客群：{fields['target_audience']}"
            ),
            brand_style=fields["brand_style"],
            promotion_goal=fields["promotion_goal"],
        )

    except Exception as exc:
        print(f"❌ [商家預覽] 生成失敗：{exc}")

        raise HTTPException(
            status_code=500,
            detail="商家預覽生成失敗，請查看後端日誌。",
        ) from exc

    return {
        "status": "success",
        "user_type": "merchant",
        "merchant_name": fields["merchant_name"],
        "retrieved_data": {
            "id": db_info.get("id") or "",
            "name": db_name,
            "description": db_description,
            "address": db_info.get("address") or "",
        },
        "draft_preview": {
            "suggested_script": draft["tw_script"],
            "promo_copy": draft["promo_copy"],
            "seo_keywords": draft["seo_keywords"],
            "music_suggestion": music_suggestion,
            "review_notes": draft["review_notes"],
            "generation_source": draft["generation_source"],
        },
        "data_sources": source_records,
        "warnings": warnings_list,
        "requires_review": True,
    }


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

# ============================================================
# 自然語言劇本藍圖
# 統一使用 job_queue，不再使用 BLUEPRINT_JOBS / BackgroundTasks
# ============================================================

class ScriptTextPromptRequest(BaseModel):
    user_prompt: str = Field(
        default="我們想要兩個人去看雲海",
        min_length=1,
        max_length=3000,
        description="使用者輸入的自然語言或語音轉文字內容",
        json_schema_extra={
            "example": "我們想要兩個人去看雲海"
        },
    )


@router.post(
    "/api/admin/generate_script_blueprint_by_text",
    summary="【自動故事 v2】需求引導、候選審核與劇本生成",
    operation_id="generate_story_from_text_auto_v2",
    status_code=202,
)
async def api_generate_script_blueprint_by_text(
    req: ScriptTextPromptRequest,
    request: Request,
):
    user_prompt = req.user_prompt.strip()

    task_id = uuid.uuid4().hex

    check_url = str(
        request.app.url_path_for(
            "get_ai_job",
            task_id=task_id,
        )
    )

    print(
        "[AUTO-STORY-V2] 接收請求"
        f"｜task_id={task_id}"
        f"｜routes={__file__}"
    )

    async def execute_job():
        job_queue.stage(
            task_id,
            "正在分析需求並查詢可用內容",
        )

        try:
            generated = await asyncio.to_thread(
                generate_story_workflow,
                user_prompt,
            )

            validated = StoryGenerationResult.model_validate(
                generated
            )

        except Exception:
            logger.exception(
                "[AUTO-STORY-V2] 未預期錯誤"
                "｜task_id=%s",
                task_id,
            )

            validated = StoryGenerationResult.model_validate(
                internal_error_result()
            )

        result = validated.model_dump(
            mode="json",
            exclude_unset=True,
        )

        story = validated.story_payload

        if story is not None:
            print(
                "[AUTO-STORY-V2] 劇本結果"
                f"｜task_id={task_id}"
                f"｜outcome={validated.outcome}"
                f"｜節點數={len(story.nodes)}"
                f"｜任務數="
                f"{sum(len(node.tasks) for node in story.nodes)}"
            )
        else:
            print(
                "[AUTO-STORY-V2] 非劇本結果"
                f"｜task_id={task_id}"
                f"｜outcome={validated.outcome}"
                f"｜reason_code={validated.reason_code}"
            )

        return result

    submitted = job_queue.submit(
        task_type="script_blueprint",
        factory=execute_job,
        task_id=task_id,
    )

    if isinstance(submitted, dict):
        return {
            **submitted,
            "job_id": task_id,
            "check_url": check_url,
            "pipeline_version": "auto-story-v2",
        }

    return submitted



@router.get(
    "/api/admin/generate_script_blueprint_by_text/{job_id}",
    summary="查詢劇本藍圖任務（相容舊路由）",
)
async def api_get_blueprint_job(job_id: str):
    return job_queue.get(job_id)

# ============================================================
# 劇本任務生成 API：批次故事排隊生成
#
# POST：驗證輸入、提交任務，回傳 HTTP 202
# GET /api/jobs/{task_id}：查詢進度與結果
# ============================================================
@router.post(
    "/stories/tasks",
    summary="生成一套故事與每站全部指定任務",
    status_code=202,
)
async def create_single_story_tasks(
    payload: PublicStoryTaskRequest,
):
    task_id = uuid.uuid4().hex

    request_data = payload.model_dump(
        mode="json",
    )

    places = request_data["places"]

    total_tasks = sum(
        len(place["type_list"])
        for place in places
    )

    print(
        f"[任務 {task_id}] "
        f"收到 {len(places)} 個景點，"
        f"共 {total_tasks} 個指定任務"
    )

    for index, place in enumerate(places, start=1):
        print(
            f"[任務 {task_id}] "
            f"第 {index} 站"
            f"｜名稱：{place['p_name']}"
            f"｜ID：{place['place_id']}"
            f"｜題型："
            f"{[item['type_id'] for item in place['type_list']]}"
        )

    async def execute_job():
        job_queue.stage(
            task_id,
            "正在生成故事、章節與各站指定任務",
        )

        generated = await asyncio.to_thread(
            generate_public_story_tasks,
            request_data,
        )

        job_queue.stage(
            task_id,
            "正在驗證公開回應格式",
        )

        validated = PublicStoryTaskResponse.model_validate(
            generated
        )

        result = validated.model_dump(
            mode="json",
            exclude_unset=True,
        )

        print(
            f"[任務 {task_id}] "
            f"完成 {len(result['nodes'])} 個節點，"
            f"共 {sum(len(node['tasks']) for node in result['nodes'])} 個任務"
        )

        return result

    return job_queue.submit(
        task_type="stories_single",
        factory=execute_job,
        task_id=task_id,
    )
import asyncio
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from core.config import COMFY_INPUT_DIR, COMFY_URL, OUTPUT_DIR
from services.llm_service import _get_llm
from services.neo4j_rag_service import search_neo4j_rag

# ---------------------------------------------------------------------------
# 路徑設定 (跨 Windows / Linux)
# ---------------------------------------------------------------------------

# 素材圖片資料夾 (npc_head.png / poastcard_stamp.png)
# 優先使用環境變數 PLAYTAIWAN_IMG_DIR
_IMG_DIR_CANDIDATES = [
    os.environ.get("PLAYTAIWAN_IMG_DIR", ""),
    r"D:\playtaiwan\playtaiwan_img",
]


def _find_img_dir():
    for d in _IMG_DIR_CANDIDATES:
        if d and os.path.isdir(d):
            return Path(d)
    return None


IMG_DIR = _find_img_dir()

# 中文字型候選 (Windows 優先，其次 Linux)
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msjh.ttc",      # 微軟正黑體
    r"C:\Windows\Fonts\msjhbd.ttc",    # 微軟正黑體 粗體
    r"C:\Windows\Fonts\mingliu.ttc",   # 細明體
    r"C:\Windows\Fonts\kaiu.ttf",      # 標楷體
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
]


def _load_workflow() -> dict:
    """在幾個常見位置尋找 postcard_workflow.json"""
    here = Path(__file__).resolve().parent
    candidates = [
        Path("postcard_workflow.json"),          # 目前工作目錄
        here / "postcard_workflow.json",         # services/
        here.parent / "postcard_workflow.json",  # 專案根目錄
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"找不到 postcard_workflow.json，已嘗試: {[str(c) for c in candidates]}")


# ---------------------------------------------------------------------------
# 文字生成
# ---------------------------------------------------------------------------

def generate_postcard_text(spot_name: str, user_prompt: str = "") -> str:
    llm = _get_llm()
    rag_context = search_neo4j_rag(spot_name, is_night_mode=False)

    system_prompt = (
        f"你是一位頂尖的文青散文詩人與在地導覽員。\n"
        f"景點名稱：{spot_name}\n"
        f"在地背景參考：{rag_context[:300]}\n\n"
        f"【絕對指令】：請撰寫一段精緻、充滿溫度且具有文青詩意的明信片短文。\n"
        f"1. 必須 100% 使用純正的繁體中文 (zh-TW) 書寫。\n"
        f"2. 嚴禁出現英文長篇大論或生硬的導覽口吻，要像寫給知己的旅途手札。\n"
        f"3. 字數限制在 80-100 字以內，不輸出多餘的說明文字或標籤。"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "請為此景點生成繁體中文明信片短文："},
    ]
    try:
        outputs = llm(messages, max_new_tokens=200, temperature=0.7, do_sample=True)
        return outputs[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        print(f"❌ LLM 生成文字失敗: {e}")
        return f"駐足於 {spot_name} 的微光中，歲月彷彿放慢了腳步，每一磚一瓦都訴說著動人的在地記憶。"


# ---------------------------------------------------------------------------
# 圖片生成
# ---------------------------------------------------------------------------

async def generate_ai_postcard_image(
    user_img_path: str, spot_name: str, user_prompt: str, task_id: str = ""
) -> str:
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    if not task_id:
        task_id = str(uuid.uuid4())[:8]

    raw_ai_output_path = os.path.join(OUTPUT_DIR, f"ai_raw_{task_id}.png")
    output_path = os.path.join(OUTPUT_DIR, f"ai_postcard_{task_id}.png")

    try:
        if COMFY_INPUT_DIR is None:
            raise RuntimeError("找不到 ComfyUI 資料夾，請設定環境變數 COMFY_DIR")

        compiled_prompt = (
            f"masterpiece, minimalist editorial illustration of {spot_name}, distinct 8-tiered stacked architecture, "
            f"exhibition catalog art style, generous negative space, pure ivory background, clean empty sky, "
            f"muted low-saturation Morandi palette, flat pastel gouache brushstrokes, 2D flat design, "
            f"exact architectural structure, sharp straight lines, simple elegant geometric shapes."
        )

        negative_prompt = (
            f"photorealistic, realistic photography, realistic lighting, 3d render, perspective, "
            f"complex textures, shading, gradients, dark, grey, mountains, clouds, messy background, "
            f"magazine cover, frame, border, text, watermark, noisy, cluttered, melted, deformed"
        )

        # 1. 產生線稿並存進 ComfyUI 的 input 資料夾
        original_pil = Image.open(user_img_path).convert("RGB")
        edges_pil = original_pil.convert("L").filter(ImageFilter.FIND_EDGES)
        edges_pil = ImageOps.autocontrast(edges_pil)

        filename_only = f"postcard_lineart_{task_id}.jpg"
        os.makedirs(str(COMFY_INPUT_DIR), exist_ok=True)
        edges_path = os.path.join(str(COMFY_INPUT_DIR), filename_only)
        edges_pil.save(edges_path)

        # 2. 載入並調整 workflow
        workflow = _load_workflow()

        # 橫向比例 (寬 768, 高 512)，產出橫式長方形圖片
        workflow["14"] = {
            "inputs": {"width": 768, "height": 512, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        }

        if "3" in workflow:
            workflow["3"]["inputs"]["latent_image"] = ["14", 0]
            workflow["3"]["inputs"]["positive"] = ["13", 0]
            workflow["3"]["inputs"]["denoise"] = 1.0

        if "13" in workflow:
            workflow["13"]["inputs"]["strength"] = 0.85

        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = compiled_prompt
        if "7" in workflow:
            workflow["7"]["inputs"]["text"] = negative_prompt
        if "10" in workflow:
            workflow["10"]["inputs"]["image"] = filename_only

        # 3. 送出到 ComfyUI
        req = urllib.request.Request(
            f"{COMFY_URL}/prompt",
            data=json.dumps({"prompt": workflow}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as response:
                prompt_id = json.loads(response.read().decode("utf-8"))["prompt_id"]
        except urllib.error.HTTPError as e:
            # ComfyUI 回 400 時，body 會說明是哪個節點/模型有問題 (例如 missing node type)
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ComfyUI 拒絕 workflow (HTTP {e.code}): {body[:800]}")

        # 4. 輪詢結果 (用 asyncio.sleep，不會卡住整個 FastAPI 事件迴圈)
        image_data = None
        comfy_error = None
        for _ in range(600):
            await asyncio.sleep(1)
            try:
                with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as hist_res:
                    history_data = json.loads(hist_res.read().decode("utf-8"))
            except Exception:
                continue

            entry = history_data.get(prompt_id)
            if not entry:
                continue

            if entry.get("status", {}).get("status_str") == "error":
                comfy_error = "ComfyUI 執行時發生錯誤，請查看 ComfyUI 終端機或 comfy_autostart.log"
                break

            for node_output in entry.get("outputs", {}).values():
                if "images" in node_output:
                    img_info = node_output["images"][0]
                    img_url = (
                        f"{COMFY_URL}/view?filename={urllib.parse.quote(img_info['filename'])}"
                        f"&subfolder={urllib.parse.quote(img_info['subfolder'])}&type={img_info['type']}"
                    )
                    with urllib.request.urlopen(img_url) as img_resp:
                        image_data = img_resp.read()
            break

        if comfy_error:
            raise RuntimeError(comfy_error)
        if not image_data:
            raise RuntimeError("ComfyUI 渲染失敗或逾時")

        with open(raw_ai_output_path, "wb") as f:
            f.write(image_data)

        # 5. PIL 後處理 (橫式 ibon 4x6 規格：1800 x 1200)
        img = Image.open(raw_ai_output_path).convert("RGB")

        base_width = 1400
        h_size = int(float(img.size[1]) * (base_width / float(img.size[0])))
        img = img.resize((base_width, h_size), Image.Resampling.LANCZOS)

        canvas_width = 1800
        canvas_height = 1200
        canvas = Image.new("RGB", (canvas_width, canvas_height), color="#FBF9F5")

        img_x = (canvas_width - base_width) // 2
        img_y = 70
        canvas.paste(img, (img_x, img_y))

        # 左下角 NPC 頭像
        if IMG_DIR:
            npc_head_path = IMG_DIR / "npc_head.png"
            if npc_head_path.exists():
                try:
                    npc_img = Image.open(npc_head_path).convert("RGBA")
                    npc_img = npc_img.resize((100, 100), Image.Resampling.LANCZOS)
                    canvas.paste(npc_img, (80, canvas_height - 130), npc_img)
                except Exception as e:
                    print(f"⚠️ 載入 NPC 頭像失敗: {e}")
            else:
                print(f"⚠️ 找不到 NPC 頭像: {npc_head_path}")

            # 右下角紀念戳章
            stamp_path = IMG_DIR / "poastcard_stamp.png"
            if stamp_path.exists():
                try:
                    stamp_img = Image.open(stamp_path).convert("RGBA")
                    stamp_size = 210
                    stamp_img = stamp_img.resize((stamp_size, stamp_size), Image.Resampling.LANCZOS)

                    stamp_x = img_x + base_width - stamp_size + 45
                    stamp_y = img_y + h_size - stamp_size + 45

                    canvas.paste(stamp_img, (stamp_x, stamp_y), stamp_img)
                except Exception as e:
                    print(f"⚠️ 載入郵戳圖片失敗: {e}")
            else:
                print(f"⚠️ 找不到郵戳圖片: {stamp_path}")
        else:
            print(f"⚠️ 找不到素材資料夾，已嘗試: {[d for d in _IMG_DIR_CANDIDATES if d]}")

        # 6. 繪製文字 (標題與製造日期)
        draw = ImageDraw.Draw(canvas)
        font = None
        small_font = None

        for f_path in _FONT_CANDIDATES:
            if os.path.exists(f_path):
                try:
                    font = ImageFont.truetype(f_path, 34)
                    small_font = ImageFont.truetype(f_path, 22)
                    break
                except Exception:
                    continue
        if font is None:
            print("⚠️ 找不到中文字型，標題可能顯示為方塊")
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        title_text = f"{spot_name}  —  日遊所思・在地明信片"
        draw.text((200, canvas_height - 115), title_text, fill="#4A4A4A", font=font)

        current_date_str = datetime.now().strftime("%Y.%m.%d")
        date_text = f"ISSUE DATE : {current_date_str}"
        draw.text((200, canvas_height - 75), date_text, fill="#888888", font=small_font)

        canvas.save(output_path, quality=95)
        print(f"✅ [Landscape Perfect Ready] 成功生成橫式滿版大器明信片: {output_path}")
        return output_path

    except Exception as e:
        print(f"⚠️ 完整管線錯誤: {e}")
        shutil.copy(user_img_path, output_path)
        return output_path
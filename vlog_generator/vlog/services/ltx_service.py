import os
import torch
import uuid
import traceback
from PIL import Image
from diffusers import LTXImageToVideoPipeline
from diffusers.utils import export_to_video
from core.config import OUTPUT_DIR

# 全域變數：確保伺服器啟動期間，模型只載入一次
_pipeline = None

def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        print("🚀 正在載入 LTX-Video AI 模型 (需消耗較多 VRAM)...")
        _pipeline = LTXImageToVideoPipeline.from_pretrained(
            "Lightricks/LTX-Video",
            torch_dtype=torch.bfloat16
        )
        _pipeline.enable_model_cpu_offload()
        print("✅ LTX-Video 模型載入完成！")
    return _pipeline

def generate_ltx_video(image_path: str, prompt: str) -> str:
    """
    實作 LTX-Video 圖生影片 (I2V) 的推論邏輯，並加上嚴格的檔案落地檢查
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到來源圖片: {image_path}")

    image = Image.open(image_path).convert("RGB")
    pipe = _get_pipeline()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    output_filename = f"ltx_{unique_id}.mp4"
    output_path = OUTPUT_DIR / output_filename

    print(f"🎥 開始生成 AI 動態影片... (提示詞: {prompt})")

    try:
        # 執行推論
        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted, watermark",
            width=768,
            height=512,
            num_frames=49,
            num_inference_steps=40,
            guidance_scale=3.0,
        )
        
        if not hasattr(output, "frames") or not output.frames:
            raise RuntimeError("LTX-Video 模型推論完成，但未回傳任何 frames 資料！")

        video_frames = output.frames[0]

        # 輸出成影片
        export_to_video(video_frames, str(output_path), fps=24)

        # 嚴格驗證檔案是否真實存在且大於 0 bytes
        resolved_path = output_path.resolve()
        if resolved_path.exists() and resolved_path.stat().st_size > 0:
            print(f"✅ AI 影片生成完畢且確認落地！大小: {resolved_path.stat().st_size} bytes, 路徑: {resolved_path}")
            return str(resolved_path)
        else:
            raise RuntimeError(f"export_to_video 執行結束，但檔案未成功寫入硬碟: {resolved_path}")

    except Exception as e:
        print(f"❌ LTX-Video 片段生成崩潰:")
        traceback.print_exc()
        raise e

import time

print("⏳ 1. 開始匯入 diffusers 與 torch 套件...")
start_time = time.time()
from diffusers import LTXImageToVideoPipeline
import torch

print("⏳ 2. 準備從硬碟讀取 17GB 的 LTX-Video 模型...")
print("   (這個步驟大約需要 30 秒 ~ 1 分鐘，請耐心等待)")
pipeline = LTXImageToVideoPipeline.from_pretrained(
    "Lightricks/LTX-Video", 
    torch_dtype=torch.bfloat16
)

end_time = time.time()
print(f"✅ 3. 模型載入成功！總共耗時: {end_time - start_time:.1f} 秒")

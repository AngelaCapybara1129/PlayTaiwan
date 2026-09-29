import os
import soundfile as sf
from datasets import load_dataset

print("⏳ 正在連線至 Hugging Face 讀取 zh-taiwan 資料集...")

# 💡 修正：在第二個參數補上 config name "train"
dataset = load_dataset("ivanzhu109/zh-taiwan", "train", split="train", streaming=True)
sample = next(iter(dataset))

# 取得音訊數據與取樣率
audio_data = sample['audio']['array']
sample_rate = sample['audio']['sampling_rate']
text = sample['text']

# 確保存檔目錄存在
os.makedirs("assets/audio", exist_ok=True)
output_path = "assets/audio/tw_reference.wav"

# 儲存為 WAV 檔
sf.write(output_path, audio_data, sample_rate)

print("✅ 下載成功！")
print(f"📁 音檔已儲存至：{output_path}")
print(f"🗣️ 這段參考音檔原本說的台詞是：\n「{text}」")

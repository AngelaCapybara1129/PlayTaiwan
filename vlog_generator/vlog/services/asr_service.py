import os
import torch
import warnings
from transformers import pipeline

# 🪄 關閉 transformers 不影響功能的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_asr_pipeline = None

def _get_asr_pipeline():
    global _asr_pipeline
    if _asr_pipeline is None:
        print("🚀 正在從 Hugging Face 載入 Whisper 多國語言辨識模型...")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            torch_dtype=torch_dtype,
            device=device,
            chunk_length_s=30,  # 🆕 告訴模型：自動把長語音切成每 30 秒一段來處理！
        )
        print("✅ Whisper ASR 模型載入完成！")
        
    return _asr_pipeline

def transcribe_audio(audio_path: str, language: str = None) -> str:
    """
    將使用者的錄音檔轉換為純文字 (支援多國語言自動偵測與超過 30 秒的長語音)
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到錄音檔: {audio_path}")

    asr_pipe = _get_asr_pipeline()

    print(f"👂 系統正在跨國語言聆聽模式... (檔案: {audio_path})")

    # 動態設定參數
    gen_kwargs = {
        "task": "transcribe",
        "return_timestamps": True  # 🆕 告訴模型：長語音處理時必須紀錄時間軸！
    }
    
    if language:
        gen_kwargs["language"] = language

    # 執行語音辨識
    result = asr_pipe(
        audio_path,
        generate_kwargs=gen_kwargs
    )
    
    transcribed_text = result["text"].strip()
    print(f"✅ 辨識完成：\n「{transcribed_text}」\n")
    
    return transcribed_text

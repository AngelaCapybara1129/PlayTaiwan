import os
import torch
import warnings
from transformers import pipeline

warnings.filterwarnings("ignore", category=UserWarning)

_ser_pipeline = None

def _get_ser_pipeline():
    global _ser_pipeline
    if _ser_pipeline is None:
        print("🚀 正在載入 SER 語音情感分析模型 (Wav2Vec2)...")
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # 載入輕量級語音情感分類模型
        _ser_pipeline = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er",
            device=device
        )
        print("✅ SER 模型載入完成！")
        
    return _ser_pipeline

def analyze_emotion(audio_path: str) -> str:
    """
    分析錄音檔中的情緒，回傳對應的中文情緒描述，供 LLM 參考
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"找不到錄音檔: {audio_path}")

    print(f"💓 系統正在感受語音中的情緒... (檔案: {audio_path})")
    
    ser_pipe = _get_ser_pipeline()
    
    # 執行音訊分類
    results = ser_pipe(audio_path)
    
    # 取信心分數最高的情感標籤 (例如：'hap', 'neu', 'sad', 'ang')
    top_emotion_label = results[0]['label']
    
    # 將英文縮寫轉換為適合 Prompt 的中文情緒設定
    emotion_map = {
        "hap": "開心、興奮、充滿活力",
        "neu": "平靜、放鬆、悠閒",
        "sad": "感性、懷舊、深沉",
        "ang": "激動、強烈" # 實務上觀光導覽較少用到生氣，可轉化為情緒激昂
    }
    
    final_emotion = emotion_map.get(top_emotion_label, "平靜、放鬆")
    print(f"✅ 情感分析完成：使用者聽起來非常【{final_emotion}】！")
    
    return final_emotion

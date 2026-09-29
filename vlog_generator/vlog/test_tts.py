from services.tts_service import generate_tts

vlog_script = "歡迎來到南投的美麗茶園。在這裡，你可以聞到清新的茶香，感受到微風拂過葉片的溫柔，這就是數位創新帶來的全新觀光體驗。"
reference_audio = "assets/audio/07071049.wav"

print("啟動台灣版 TTS 測試...")
try:
    # 改用 zh-tw 代碼
    output_audio_path = generate_tts(
        text=vlog_script, 
        reference_audio_path=reference_audio, 
        language="zh-cn" 
    )
    
    print(f"🎉 測試成功！快去聽聽看濃濃台灣味的檔案：{output_audio_path}")
except Exception as e:
    print(f"❌ 測試失敗：{e}")

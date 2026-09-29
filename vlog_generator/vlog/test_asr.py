from services.asr_service import transcribe_audio

# 假設這是使用者在南投茶園用手機錄下的一段語音
user_audio_input = "assets/audio/tw_reference.wav"

print("啟動 ASR 聽覺測試...")
try:
    # 呼叫 ASR 服務將語音轉成文字
    text_result = transcribe_audio(user_audio_input)
    
    print("🎉 測試成功！系統已經聽懂了這句話：")
    print(f"➡️ {text_result}")
    
except Exception as e:
    print(f"❌ 測試失敗：{e}")

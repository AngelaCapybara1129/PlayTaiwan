from services.llm_service import generate_vlog_content

# 這是我們剛剛 ASR 辨識出來的「瑕疵」草稿
raw_asr_text = "這可能是第一個Firefox Unenjoyed的版本Credit。"

print("啟動 LLM 大腦測試...")
try:
    result = generate_vlog_content(raw_asr_text)
    
    print("\n🎉 測試成功！看看大腦的轉換結果：")
    print("-" * 40)
    print(f"🎙️ 給配音員的旁白 (tw_script): \n{result['tw_script']}")
    print("-" * 40)
    print(f"🎬 給影片 AI 的提示詞 (en_video_prompt): \n{result['en_video_prompt']}")
    print("-" * 40)
    
except Exception as e:
    print(f"❌ 測試失敗：{e}")

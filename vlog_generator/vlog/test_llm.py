from services.llm_service import generate_vlog_content_with_template, preload_llm

raw_asr_text = "這可能是第一個Firefox Unenjoyed的版本Credit。"

print("啟動 LLM 大腦測試...")
try:
    preload_llm()
    result = generate_vlog_content_with_template(
        raw_text=raw_asr_text,
        emotion="開心",
        template="user_vlog",
    )
    print("\n🎉 測試成功！")
    print("-" * 40)
    print(f"🎙️ 旁白 (tw_script):\n{result['tw_script']}")
    print("-" * 40)
    print(f"🎬 影片提示詞 (en_video_prompt):\n{result['en_video_prompt']}")
    print("-" * 40)
except Exception as e:
    print(f"❌ 測試失敗：{e}")
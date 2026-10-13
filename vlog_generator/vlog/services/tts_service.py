import os
import uuid
import asyncio
import requests
from core.config import OUTPUT_DIR

# 直接接收 npc_name (例如 "阿達力")
async def generate_tts(text: str, npc_name: str = "薯光"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    
    output_filename = f"tts_{unique_id}.wav"
    vtt_filename = f"tts_{unique_id}.vtt"
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    vtt_path = os.path.join(OUTPUT_DIR, vtt_filename)

    print(f"🎙️ 開始生成 XTTS 語音... \nNPC: {npc_name}\n台詞: {text}")

    # 直接組合出音檔名稱交給 XTTS
    speaker_filename = f"{npc_name}.wav"

    def _generate_xtts():
        url = "http://localhost:8020/tts_to_audio/"
        
        payload = {
            "text": text,
            "language": "zh-cn",
            "speaker_wav": speaker_filename
        }
        
        response = requests.post(url, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"XTTS API 錯誤 ({response.status_code}): {response.text}")
        
        with open(output_path, 'wb') as audio_file:
            audio_file.write(response.content)

    await asyncio.to_thread(_generate_xtts)
    
    vtt_lines = [
        "WEBVTT\n\n",
        "00:00:00.000 --> 00:59:59.000\n",
        f"{text}\n\n"
    ]
    with open(vtt_path, "w", encoding="utf-8") as file:
        file.writelines(vtt_lines)

    return str(output_path), str(vtt_path)
import os
import uuid
import edge_tts
from core.config import OUTPUT_DIR

def format_vtt_time(ticks: int) -> str:
    """將 Edge-TTS 的 100 奈秒 (ticks) 轉換為 VTT 支援的 HH:MM:SS.mmm 格式"""
    seconds = ticks / 10_000_000
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{s:06.3f}"

async def generate_tts(text: str, voice: str = "zh-TW-HsiaoChenNeural"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    unique_id = uuid.uuid4().hex[:8]
    output_filename = f"tts_{unique_id}.mp3"
    vtt_filename = f"tts_{unique_id}.vtt"
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    vtt_path = os.path.join(OUTPUT_DIR, vtt_filename)

    if len(text) > 150:
        print(f"⚠️ 警告：旁白略長 ({len(text)} 字)，進行自動精簡...")
        cut_text = text[:150]
        if "。" in cut_text:
            text = cut_text.rsplit("。", 1)[0] + "。"
        elif "，" in cut_text:
            text = cut_text.rsplit("，", 1)[0] + "..."
        else:
            text = cut_text + "..."

    print(f"🎙️ 開始生成台灣 AI 旁白 (Edge-TTS) 並提取時間軸... \n實際唸稿: {text}")

    communicate = edge_tts.Communicate(text, voice)
    
    # 準備 VTT 字幕的開頭
    vtt_lines = ["WEBVTT\n\n"]

    # 串流寫入音檔，並同時攔截每個字的精準時間
    with open(output_path, "wb") as file:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                file.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start_time = format_vtt_time(chunk["offset"])
                end_time = format_vtt_time(chunk["offset"] + chunk["duration"])
                # 將字詞時間軸加入 VTT 列表
                vtt_lines.append(f"{start_time} --> {end_time}\n{chunk['text']}\n\n")

    # 寫入 VTT 檔案
    with open(vtt_path, "w", encoding="utf-8") as file:
        file.writelines(vtt_lines)

    print(f"✅ AI 旁白與字幕生成完畢！\n音檔儲存至: {output_path}\n字幕儲存至: {vtt_path}")
    
    return str(output_path), str(vtt_path)

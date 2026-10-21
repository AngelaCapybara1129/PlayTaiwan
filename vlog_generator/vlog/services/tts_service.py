import asyncio
import os
import re
import subprocess
import tempfile
import threading
import uuid
import wave
from pathlib import Path

import requests

from core.config import OUTPUT_DIR
from services.moviepy_service import find_exe


XTTS_URL = "http://127.0.0.1:8020/tts_to_audio/"

# 先採保守長度，配合你目前直式字幕的顯示寬度。
# 太短會增加 API 次數，也可能降低語氣連貫性。
TTS_MAX_CHARS = 16

OUTPUT_SAMPLE_RATE = 24000
SENTENCE_GAP_SECONDS = 0.12

# 同一個應用程序內，避免多個請求交錯呼叫 XTTS。
# 不會控制其他程序或其他服務直接送到 XTTS 的請求。
_TTS_LOCK = threading.Lock()


def split_tts_sentences(text: str, max_chars: int = TTS_MAX_CHARS):
    """優先依標點切句；過長片段再切短，保留標點。"""
    text = re.sub(r"\s+", " ", text).strip()

    parts = re.findall(
        r"[^，。！？；、,.!?;]+[，。！？；、,.!?;]*",
        text,
    )

    chunks = []

    for part in parts:
        part = part.strip()

        while len(part) > max_chars:
            chunks.append(part[:max_chars])
            part = part[max_chars:].strip()

        if part:
            chunks.append(part)

    # 只有標點、沒有可朗讀內容時，不送入 XTTS。
    return [
        chunk
        for chunk in chunks
        if re.search(r"[^\W_]", chunk, flags=re.UNICODE)
    ]


def format_vtt_time(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))

    hours, remaining = divmod(milliseconds, 3_600_000)
    minutes, remaining = divmod(remaining, 60_000)
    seconds_int, milliseconds = divmod(remaining, 1000)

    return (
        f"{hours:02d}:{minutes:02d}:{seconds_int:02d}"
        f".{milliseconds:03d}"
    )


def request_xtts_audio(session, text: str, speaker_filename: str) -> bytes:
    response = session.post(
        XTTS_URL,
        json={
            "text": text,
            "language": "zh-cn",
            "speaker_wav": speaker_filename,
        },
        timeout=(10, 600),
    )

    content_type = response.headers.get(
        "Content-Type", ""
    ).lower()

    if response.status_code != 200:
        raise RuntimeError(
            f"XTTS API 錯誤 ({response.status_code})："
            f"{response.text[:1000]}"
        )

    if "json" in content_type or "text/html" in content_type:
        raise RuntimeError(
            f"XTTS 沒有回傳預期音訊：{response.text[:1000]}"
        )

    audio_bytes = response.content

    if (
        len(audio_bytes) < 12
        or audio_bytes[:4] not in (b"RIFF", b"RF64")
        or audio_bytes[8:12] != b"WAVE"
    ):
        raise RuntimeError(
            "XTTS 回傳內容不是預期的 WAV："
            f"type={content_type}, "
            f"header={audio_bytes[:32]!r}"
        )

    return audio_bytes


def normalize_wav(source_path: Path, target_path: Path):
    result = subprocess.run(
        [
            find_exe("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(source_path),
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(OUTPUT_SAMPLE_RATE),
            "-c:a", "pcm_s16le",
            str(target_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "TTS 音訊格式轉換失敗："
            + result.stderr[-1500:]
        )


async def generate_tts(text: str, npc_name: str = "薯光"):
    """分句生成 WAV，並建立依實際音訊長度計算的 VTT。"""
    text = text.strip()

    if not text:
        raise ValueError("TTS 台詞不可為空白")

    # speaker_wav 僅接受角色檔名，不接受任意路徑。
    if (
        not npc_name
        or any(char in npc_name for char in "/\\:")
        or npc_name in {".", ".."}
    ):
        raise ValueError("無效的 NPC 名稱")

    chunks = split_tts_sentences(text)

    if not chunks:
        raise ValueError("TTS 台詞沒有可朗讀的內容")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_id = uuid.uuid4().hex
    output_path = output_dir / f"tts_{unique_id}.wav"
    vtt_path = output_dir / f"tts_{unique_id}.vtt"
    building_path = output_dir / f"tts_{unique_id}_building.wav"

    speaker_filename = f"{npc_name}.wav"

    def generate_all():
        with _TTS_LOCK:
            cues = []
            frame_cursor = 0

            gap_frames = round(
                SENTENCE_GAP_SECONDS * OUTPUT_SAMPLE_RATE
            )
            silence = b"\x00\x00" * gap_frames

            try:
                with (
                    tempfile.TemporaryDirectory(
                        prefix="tts_segments_",
                        dir=str(output_dir),
                    ) as temp_dir,
                    requests.Session() as session,
                    wave.open(str(building_path), "wb") as merged,
                ):
                    merged.setnchannels(1)
                    merged.setsampwidth(2)
                    merged.setframerate(OUTPUT_SAMPLE_RATE)

                    for index, sentence in enumerate(chunks):
                        print(
                            f"🎙️ XTTS {index + 1}/{len(chunks)}"
                            f"｜NPC：{npc_name}｜{sentence}"
                        )

                        raw_path = Path(temp_dir) / f"{index}_raw.wav"
                        pcm_path = Path(temp_dir) / f"{index}_pcm.wav"

                        audio_bytes = request_xtts_audio(
                            session,
                            sentence,
                            speaker_filename,
                        )

                        raw_path.write_bytes(audio_bytes)
                        normalize_wav(raw_path, pcm_path)

                        with wave.open(str(pcm_path), "rb") as clip:
                            frame_count = clip.getnframes()

                            if frame_count <= 0:
                                raise RuntimeError(
                                    f"第 {index + 1} 句沒有有效音訊"
                                )

                            audio_data = clip.readframes(frame_count)

                        start_t = frame_cursor / OUTPUT_SAMPLE_RATE

                        merged.writeframesraw(audio_data)
                        frame_cursor += frame_count

                        end_t = frame_cursor / OUTPUT_SAMPLE_RATE

                        cues.append((start_t, end_t, sentence))

                        if index < len(chunks) - 1:
                            merged.writeframesraw(silence)
                            frame_cursor += gap_frames

                # with 結束後 WAV 標頭才已完整寫入。
                os.replace(building_path, output_path)

                vtt_lines = ["WEBVTT\n\n"]

                for index, (start_t, end_t, sentence) in enumerate(
                    cues,
                    start=1,
                ):
                    vtt_lines.extend([
                        f"{index}\n",
                        (
                            f"{format_vtt_time(start_t)} --> "
                            f"{format_vtt_time(end_t)}\n"
                        ),
                        f"{sentence}\n\n",
                    ])

                vtt_path.write_text(
                    "".join(vtt_lines),
                    encoding="utf-8",
                )

                print(
                    f"✅ TTS 完成：{len(cues)} 段，"
                    f"{frame_cursor / OUTPUT_SAMPLE_RATE:.2f}s"
                )

            except Exception:
                for path in (
                    building_path,
                    output_path,
                    vtt_path,
                ):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                raise

    await asyncio.to_thread(generate_all)

    return (
        str(output_path.resolve()),
        str(vtt_path.resolve()),
    )
import glob
import os
import re
import shutil
import subprocess
import traceback
from functools import lru_cache
from pathlib import Path

from core.config import OUTPUT_DIR, TEMPLATES

# ------------------------------------------------------------------
# FFmpeg 路徑設定
# 先查 PATH，找不到再用下面的備用資料夾，最後再搜尋 WinGet 安裝目錄
# ------------------------------------------------------------------
FFMPEG_BIN_DIR = (
    r"C:\Users\User\AppData\Local\Microsoft\WinGet\Packages"
    r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
    r"\ffmpeg-9.0.2-full_build\bin"
)


@lru_cache(maxsize=None)
def find_exe(name: str) -> str:
    """尋找 ffmpeg / ffprobe 的完整路徑"""
    found = shutil.which(name)
    if found:
        return found

    candidate = os.path.join(FFMPEG_BIN_DIR, f"{name}.exe")
    if os.path.exists(candidate):
        return candidate

    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        pattern = os.path.join(
            local_appdata, "Microsoft", "WinGet", "Packages",
            "Gyan.FFmpeg*", "**", "bin", f"{name}.exe"
        )
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]

    raise FileNotFoundError(f"找不到 {name}，請確認已安裝 ffmpeg 或修改 FFMPEG_BIN_DIR")


def run_ffmpeg(cmd, task_id, step):
    """執行 ffmpeg 指令，失敗時印出真正的錯誤訊息並中止"""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        print(f"[任務 {task_id}] ❌ {step} 失敗 (returncode={result.returncode}):\n{result.stderr[-1500:]}")
        raise RuntimeError(f"ffmpeg {step} 失敗")


def get_audio_duration(file_path):
    """取得語音檔案的精準秒數，確保影片長度足夠"""
    try:
        result = subprocess.run(
            [find_exe("ffprobe"), "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"⚠️ ffprobe 獲取時間失敗: {e}")
        return 10.0  # 若失敗預設給 10 秒


def clean_subtitle_text(text: str) -> str:
    if not text:
        return ""
    # 移除 Emoji 與特殊符號，只保留中文、英文、數字、常見標點
    clean_pattern = re.compile(r"[^\u4e00-\u9fa5a-zA-Z0-9\s\-\(\)\/\:\.]+")
    cleaned = clean_pattern.sub(r"", text)
    cleaned = cleaned.replace("'", "").replace('"', "").replace("`", "")
    cleaned = cleaned.replace(":", "\\:").replace("%", "\\%")
    cleaned = " ".join(cleaned.split())
    return cleaned


def find_valid_font():
    """尋找可用的中文字型路徑（Windows / Linux）"""
    candidate_fonts = [
        "C:/Windows/Fonts/msjh.ttc",     # 微軟正黑體
        "C:/Windows/Fonts/msjhbd.ttc",
        "C:/Windows/Fonts/mingliu.ttc",  # 新細明體
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for font in candidate_fonts:
        if os.path.exists(font):
            # FFmpeg 濾鏡語法中，冒號必須跳脫
            return font.replace("\\", "/").replace(":", "\\:")
    return None


def split_subtitle_sentences(subtitle_text: str, max_len: int = 26):
    """以標點切句，過長的句子再切成小段，避免字幕超出畫面"""
    parts = re.split(r"([。！，？、；\.,!?;])", subtitle_text)
    sentences = []
    for i in range(0, len(parts) - 1, 2):
        sentences.append((parts[i] + parts[i + 1]).strip())
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())

    result = []
    for s in sentences:
        if not s:
            continue
        while len(s) > max_len:
            result.append(s[:max_len])
            s = s[max_len:]
        if s:
            result.append(s)
    return result


def process_vlog_task(
    task_id,
    image_files,
    prompt,
    tts_audio_file,
    bgm_file,
    output_file,
    template_type,
    merchant_name="",
    subtitle_text="",
    spot_metadata=None,
    vtt_file=None
):
    tpl = TEMPLATES.get(template_type, TEMPLATES["user_vlog"])
    print(f"\n[任務 {task_id}] 🚀 開始合成 (模板: {tpl['name']})")

    concat_list_path = None
    temp_video_path = None
    slide_videos = []

    try:
        if not image_files:
            raise ValueError("沒有提供任何圖片 (image_files 為空)")

        ffmpeg = find_exe("ffmpeg")

        os.makedirs(str(OUTPUT_DIR), exist_ok=True)
        final_output_path = os.path.abspath(os.path.join(str(OUTPUT_DIR), output_file))
        temp_video_path = os.path.abspath(os.path.join(str(OUTPUT_DIR), f"{task_id}_temp_merged.mp4"))
        concat_list_path = os.path.abspath(os.path.join(str(OUTPUT_DIR), f"{task_id}_list.txt"))

        font_path = find_valid_font()
        print(f"[任務 {task_id}] 🔤 使用字型路徑: {font_path}")

        # 🎵 1. 動態計算語音與照片長度
        audio_duration = get_audio_duration(tts_audio_file)
        num_images = len(image_files)
        time_per_slide = (audio_duration + 0.5) / num_images  # 加入 0.5 秒緩衝
        print(f"[任務 {task_id}] 🎵 語音長度: {audio_duration:.2f}s, 分配每張照片: {time_per_slide:.2f}s")

        fps = 25
        frames = max(int(time_per_slide * fps) + 1, 2)

        # 2. 每張照片產生一段帶有緩慢放大效果的影片
        for idx, img_path in enumerate(image_files):
            slide_out = os.path.abspath(os.path.join(str(OUTPUT_DIR), f"{task_id}_slide_{idx}.mp4"))
            slide_videos.append(slide_out)

            meta_info = spot_metadata[idx] if spot_metadata and idx < len(spot_metadata) else {}
            spot_name = meta_info.get("spot_name", merchant_name or "探索據點")
            codename = meta_info.get("location_codename", "")
            visit_time = meta_info.get("visit_time", "")

            watermark_text = f"地點: {spot_name}"
            if codename:
                watermark_text += f" ({codename})"
            if visit_time:
                watermark_text += f"  時間: {visit_time}"

            safe_wm = clean_subtitle_text(watermark_text)

            # 先補成 1920x1080 再 zoompan，可減少放大時的抖動
            # 單張圖片輸入 (不使用 -loop)，由 zoompan 的 d 決定輸出幀數
            vf_filter = (
                "scale=1920:1080:force_original_aspect_ratio=decrease,"
                "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
                f"zoompan=z='min(zoom+0.0015,1.08)'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={frames}:s=1280x720:fps={fps},"
            )

            if safe_wm and font_path:
                vf_filter += (
                    f"drawtext=fontfile='{font_path}':text='{safe_wm}':fontcolor=white:fontsize=28"
                    ":box=1:boxcolor=black@0.6:boxborderw=8:x=40:y=40,"
                )
            elif safe_wm:
                print(f"[任務 {task_id}] ⚠️ 找不到中文字型，略過地點浮水印")

            vf_filter += "format=yuv420p"

            slide_cmd = [
                ffmpeg, "-y",
                "-i", str(img_path),
                "-t", f"{time_per_slide:.3f}",
                "-vf", vf_filter,
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                slide_out,
            ]
            run_ffmpeg(slide_cmd, task_id, f"第 {idx + 1} 張幻燈片")

        # 3. 建立 FFmpeg concat 清單 (使用正斜線，避免 Windows 反斜線問題)
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for slide in slide_videos:
                f.write(f"file '{Path(slide).as_posix()}'\n")

        # 4. 串接所有幻燈片片段
        concat_cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            temp_video_path,
        ]
        run_ffmpeg(concat_cmd, task_id, "串接幻燈片")

        # 5. 準備主字幕 (以標點符號切段，按字數比例動態上字)
        video_filter_arg = "format=yuv420p"
        if subtitle_text:
            sentences = split_subtitle_sentences(subtitle_text)

            if sentences and font_path:
                total_chars = sum(len(s) for s in sentences)
                current_t = 0.0
                drawtexts = []

                for s in sentences:
                    safe_sub = clean_subtitle_text(s)
                    if not safe_sub:
                        # 沒有可顯示的字元，但仍要累加時間避免後面字幕提前
                        current_t += (len(s) / total_chars) * audio_duration
                        continue

                    duration = (len(s) / total_chars) * audio_duration
                    end_t = current_t + duration

                    dt = (
                        f"drawtext=fontfile='{font_path}':text='{safe_sub}':fontcolor=white:fontsize=36"
                        ":box=1:boxcolor=black@0.6:boxborderw=10"
                        ":x=(w-text_w)/2:y=h-120"
                        f":enable='between(t,{current_t:.3f},{end_t:.3f})'"
                    )
                    drawtexts.append(dt)
                    current_t = end_t

                if drawtexts:
                    video_filter_arg += "," + ",".join(drawtexts)
            elif sentences:
                print(f"[任務 {task_id}] ⚠️ 找不到中文字型，略過字幕")

        # 6. 最終合檔
        has_bgm = bgm_file and os.path.exists(str(bgm_file))
        if has_bgm:
            final_cmd = [
                ffmpeg, "-y",
                "-i", temp_video_path,
                "-i", str(tts_audio_file),
                "-i", str(bgm_file),
                "-filter_complex",
                "[1:a]volume=1.0[a1];[2:a]volume=0.2[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
                "-vf", video_filter_arg,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "[aout]",
                "-shortest",
                "-movflags", "+faststart",
                final_output_path,
            ]
        else:
            final_cmd = [
                ffmpeg, "-y",
                "-i", temp_video_path,
                "-i", str(tts_audio_file),
                "-vf", video_filter_arg,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                "-movflags", "+faststart",
                final_output_path,
            ]

        run_ffmpeg(final_cmd, task_id, "最終合檔")

        if not os.path.exists(final_output_path):
            raise RuntimeError("ffmpeg 執行完畢但找不到輸出檔案")

        print(f"[任務 {task_id}] ✅ 影片合成成功: {final_output_path}")

    except Exception as e:
        print(f"[任務 {task_id}] ❌ 合成崩潰: {e}")
        traceback.print_exc()

    finally:
        for p in [concat_list_path, temp_video_path] + slide_videos:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
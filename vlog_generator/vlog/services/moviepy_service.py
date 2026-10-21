import glob
import os
import re
import shutil
import subprocess
import traceback
from functools import lru_cache
from pathlib import Path
import math
import tempfile

from PIL import Image, ImageOps
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

# ------------------------------------------------------------------
# 自然旅遊 Vlog 運鏡設定
# ------------------------------------------------------------------
VLOG_MOTION = {
    # 設為 1 可降低中間畫面的記憶體需求；
    # 設為 2 會先在較高解析度運鏡，再縮回輸出尺寸。
    "supersample": 2,

    # 每張照片的最大縮放變化。
    # 建議先用 0.035，不要一開始就調得很大。
    "zoom_amount": 0.035,

    # 平移時預留的裁切餘裕。
    "pan_zoom": 1.04,

    # 模糊背景強度。
    "background_blur": 28,

    # 沒有指定 motion_type 時，依序使用這些運鏡。
    "sequence": (
        "push_in",
        "pull_out",
        "pan_right",
        "push_in",
        "pan_left",
        "pull_out",
    ),
}


def choose_motion_type(index, meta_info, duration):
    """
    可由 spot_metadata 指定：
    push_in / pull_out / pan_left / pan_right / still

    未指定時，依固定順序分配，避免每次生成效果都不同。
    """
    valid_types = {
        "push_in",
        "pull_out",
        "pan_left",
        "pan_right",
        "still",
    }

    requested = str(
        meta_info.get("motion_type") or ""
    ).strip().lower()

    if requested in valid_types:
        return requested

    # 很短的片段避免橫向移動。
    if duration < 2.0:
        return "push_in"

    sequence = VLOG_MOTION["sequence"]
    return sequence[index % len(sequence)]


def normalize_vlog_image(image_path, output_path):
    """
    處理手機照片 EXIF 旋轉、透明背景與圖片模式。
    回傳實際送進 FFmpeg 的圖片寬高。
    """
    with Image.open(str(image_path)) as source:
        image = ImageOps.exif_transpose(source).convert("RGBA")

        # 透明區域補上深色，避免直接轉 RGB 產生非預期底色。
        canvas = Image.new(
            "RGBA",
            image.size,
            (18, 18, 18, 255),
        )
        canvas.alpha_composite(image)

        normalized = canvas.convert("RGB")
        normalized.save(str(output_path), format="PNG")

        return normalized.size


def build_natural_motion_filter(
    width,
    height,
    image_width,
    image_height,
    frames,
    fps,
    motion_type,
    watermark_filter="",
):
    """
    背景：固定、填滿畫面、模糊。
    前景：保持比例，在高解析度做小幅運鏡，再縮回。
    最後才疊地點標籤。
    """
    supersample = max(
        1,
        int(VLOG_MOTION["supersample"]),
    )

    # 算出前景等比例放入輸出畫面後的尺寸。
    fit_scale = min(
        width / image_width,
        height / image_height,
    )

    foreground_width = min(
        width,
        max(2, int(image_width * fit_scale) // 2 * 2),
    )
    foreground_height = min(
        height,
        max(2, int(image_height * fit_scale) // 2 * 2),
    )

    work_width = foreground_width * supersample
    work_height = foreground_height * supersample

    # on 是 zoompan 的輸出影格編號，從 0 開始。
    progress = (
        f"min(max(on/{max(frames - 1, 1)},0),1)"
    )

    # smoothstep：起點、終點速度為 0。
    eased = (
        f"(({progress})*({progress})"
        f"*(3-2*({progress})))"
    )

    zoom_amount = float(VLOG_MOTION["zoom_amount"])
    pan_zoom = float(VLOG_MOTION["pan_zoom"])

    # 很短的片段減少動作幅度。
    duration = frames / fps
    strength = min(1.0, duration / 3.5)
    zoom_amount *= strength
    pan_zoom = 1.0 + (pan_zoom - 1.0) * strength

    center_x = "(iw-iw/zoom)/2"
    center_y = "(ih-ih/zoom)/2"

    if motion_type == "pull_out":
        zoom_expr = (
            f"1+{zoom_amount:.6f}*(1-({eased}))"
        )
        x_expr = center_x
        y_expr = center_y

    elif motion_type == "pan_right":
        zoom_expr = f"{pan_zoom:.6f}"
        x_expr = (
            f"(iw-iw/zoom)*(0.24+0.52*({eased}))"
        )
        y_expr = center_y

    elif motion_type == "pan_left":
        zoom_expr = f"{pan_zoom:.6f}"
        x_expr = (
            f"(iw-iw/zoom)*(0.76-0.52*({eased}))"
        )
        y_expr = center_y

    elif motion_type == "still":
        zoom_expr = "1"
        x_expr = "0"
        y_expr = "0"

    else:
        zoom_expr = (
            f"1+{zoom_amount:.6f}*({eased})"
        )
        x_expr = center_x
        y_expr = center_y

    blur = float(VLOG_MOTION["background_blur"])

    graph = [
        # 輸入是一張已處理旋轉的照片。
        "[0:v]setsar=1,split=2[bg_src][fg_src]",

        # 背景只處理第一張影格，再循環播放，保持固定。
        (
            "[bg_src]"
            f"scale={width}:{height}"
            ":force_original_aspect_ratio=increase"
            ":force_divisible_by=2,"
            f"crop={width}:{height},"
            f"gblur=sigma={blur:.3f},"
            "eq=brightness=-0.08:saturation=0.85,"
            "format=yuv420p,"
            "loop=loop=-1:size=1:start=0,"
            f"setpts=N/({fps}*TB)"
            "[bg]"
        ),

        # 前景先放大到工作尺寸，運鏡後再縮回。
        # 使用 yuv444p 中間格式，降低色度取樣對位移的影響。
        (
            "[fg_src]"
            f"scale={work_width}:{work_height}:flags=lanczos,"
            "format=yuv444p,"
            f"zoompan=z='{zoom_expr}'"
            f":x='{x_expr}'"
            f":y='{y_expr}'"
            f":d={frames}"
            f":s={work_width}x{work_height}"
            f":fps={fps},"
            f"scale={foreground_width}:{foreground_height}"
            ":flags=lanczos,"
            "setsar=1,"
            f"setpts=N/({fps}*TB)"
            "[fg]"
        ),
    ]

    final_filters = [
        (
            "[bg][fg]"
            "overlay=x=(W-w)/2:y=(H-h)/2"
            ":shortest=1:format=auto"
        ),
    ]

    if watermark_filter:
        final_filters.append(watermark_filter)

    final_filters.extend([
        "setsar=1",
        "format=yuv420p",
    ])

    graph.append(
        ",".join(final_filters) + "[vout]"
    )

    return ";".join(graph)

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
    vtt_file=None,
    video_orientation="landscape",
):
    concat_list_path = None
    temp_video_path = None
    building_output_path = None
    slide_videos = []
    label_text_paths = []

    try:
        tpl = TEMPLATES.get(template_type, TEMPLATES["user_vlog"])

        print(
            f"\n[任務 {task_id}] 🚀 開始合成 "
            f"(模板: {tpl['name']})"
        )

        if not image_files:
            raise ValueError("沒有提供任何圖片")

        if not tts_audio_file or not os.path.isfile(str(tts_audio_file)):
            raise FileNotFoundError(
                f"找不到 TTS 音檔：{tts_audio_file}"
            )

        profiles = {
            "portrait": {
                "width": 1080,
                "height": 1920,
                "subtitle_max_len": 16,
                "subtitle_font_size": 42,
                "watermark_font_size": 30,
                "subtitle_bottom": 300,
            },
            "landscape": {
                "width": 1920,
                "height": 1080,
                "subtitle_max_len": 26,
                "subtitle_font_size": 44,
                "watermark_font_size": 32,
                "subtitle_bottom": 150,
            },
        }

        if video_orientation not in profiles:
            raise ValueError(
                f"不支援的影片方向：{video_orientation}"
            )

        profile = profiles[video_orientation]
        width = profile["width"]
        height = profile["height"]

        ffmpeg = find_exe("ffmpeg")
        ffprobe = find_exe("ffprobe")

        os.makedirs(str(OUTPUT_DIR), exist_ok=True)

        final_output_path = os.path.abspath(
            os.path.join(str(OUTPUT_DIR), output_file)
        )
        building_output_path = os.path.abspath(
            os.path.join(str(OUTPUT_DIR), f"{task_id}_building.mp4")
        )
        temp_video_path = os.path.abspath(
            os.path.join(str(OUTPUT_DIR), f"{task_id}_temp_merged.mp4")
        )
        concat_list_path = os.path.abspath(
            os.path.join(str(OUTPUT_DIR), f"{task_id}_list.txt")
        )

        font_path = find_valid_font()

        print(
            f"[任務 {task_id}] 📐 輸出："
            f"{video_orientation}，{width}×{height}"
        )
        print(f"[任務 {task_id}] 🔤 字型：{font_path}")
        print(
            f"[任務 {task_id}] 🎙️ TTS來源："
            f"{os.path.abspath(str(tts_audio_file))}"
        )
        print(
            f"[任務 {task_id}] 🎵 BGM來源："
            f"{bgm_file or '未提供'}"
        )

        # 1. 驗證 TTS 音軌及取得時長。
        # 不使用失敗後默認 10 秒的方式，避免無效音檔被繼續合成。
        audio_probe = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(tts_audio_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if (
            audio_probe.returncode != 0
            or not audio_probe.stdout.strip()
        ):
            raise RuntimeError(
                "TTS 檔案沒有可讀取的音軌："
                + audio_probe.stderr[-1000:]
            )

        duration_probe = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(tts_audio_file),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if duration_probe.returncode != 0:
            raise RuntimeError(
                "無法讀取 TTS 時長："
                + duration_probe.stderr[-1000:]
            )

        try:
            audio_duration = float(duration_probe.stdout.strip())
        except ValueError as e:
            raise RuntimeError("TTS 時長不是有效數值") from e

        if not 0 < audio_duration < float("inf"):
            raise ValueError(f"TTS 時長無效：{audio_duration}")

        num_images = len(image_files)        
        fps = 25

        # 先換成整數影格數，避免每張片段的小數時間
        # 在串接後產生累積誤差。
        target_duration = audio_duration + 0.5

        total_frames = max(
            num_images,
            math.ceil(target_duration * fps),
        )

        base_frames, extra_frames = divmod(
            total_frames,
            num_images,
        )

        slide_frame_counts = [
            base_frames + (1 if idx < extra_frames else 0)
            for idx in range(num_images)
        ]

        average_duration = (
            total_frames / fps / num_images
        )

        print(
            f"[任務 {task_id}] 🎵 "
            f"語音長度：{audio_duration:.2f}s，"
            f"平均每張照片：{average_duration:.2f}s"
        )

        if average_duration < 1.5:
            print(
                f"[任務 {task_id}] ⚠️ "
                "目前照片數相對旁白時長偏多，"
                "每張照片的觀看時間較短。"
            )

        # 2. 每張照片生成自然微運鏡片段。
        # 保留原本 slide_videos，供後續 concat 與清理使用。
        for idx, img_path in enumerate(image_files):
            if not os.path.isfile(str(img_path)):
                raise FileNotFoundError(
                    f"找不到圖片：{img_path}"
                )

            frames = slide_frame_counts[idx]
            slide_duration = frames / fps

            slide_out = os.path.abspath(
                os.path.join(
                    str(OUTPUT_DIR),
                    f"{task_id}_slide_{idx}.mp4",
                )
            )
            slide_videos.append(slide_out)

            meta_info = (
                spot_metadata[idx]
                if spot_metadata and idx < len(spot_metadata)
                else {}
            )

            if not isinstance(meta_info, dict):
                meta_info = {}

            spot_name = (
                meta_info.get("spot_name")
                or merchant_name
                or "探索據點"
            )

            codename = (
                meta_info.get("location_codename") or ""
            )
            visit_time = (
                meta_info.get("visit_time") or ""
            )

            watermark_text = f"地點: {spot_name}"

            if codename:
                watermark_text += f" ({codename})"

            if visit_time:
                watermark_text += f"  時間: {visit_time}"

            # 避免左上角地點標籤過長。
            watermark_limit = (
                24
                if video_orientation == "portrait"
                else 44
            )
            watermark_text = watermark_text[
                :watermark_limit
            ]

            safe_wm = clean_subtitle_text(
                watermark_text
            )

            watermark_filter = ""

            if safe_wm and font_path:
                watermark_filter = (
                    f"drawtext=fontfile='{font_path}'"
                    f":text='{safe_wm}'"
                    ":expansion=none"
                    ":fontcolor=white"
                    f":fontsize={profile['watermark_font_size']}"
                    ":box=1"
                    ":boxcolor=black@0.38"
                    ":boxborderw=12"
                    ":x=40"
                    ":y=40"
                )

            motion_type = choose_motion_type(
                idx,
                meta_info,
                slide_duration,
            )

            motion_names = {
                "push_in": "緩慢推近",
                "pull_out": "緩慢拉遠",
                "pan_left": "視角向左",
                "pan_right": "視角向右",
                "still": "保留固定構圖",
            }

            print(
                f"[任務 {task_id}] 🎥 "
                f"{idx + 1}/{num_images}："
                f"{motion_names[motion_type]}，"
                f"{slide_duration:.2f}s，"
                f"{frames} frames"
            )

            # 每張照片使用獨立暫存目錄。
            # 即使 FFmpeg 出錯，也會離開 with 後自動清除。
            with tempfile.TemporaryDirectory(
                prefix="vlog_photo_",
                dir=str(OUTPUT_DIR),
            ) as temp_dir:
                normalized_image = os.path.join(
                    temp_dir,
                    "normalized.png",
                )

                image_width, image_height = (
                    normalize_vlog_image(
                        img_path,
                        normalized_image,
                    )
                )

                filter_graph = (
                    build_natural_motion_filter(
                        width=width,
                        height=height,
                        image_width=image_width,
                        image_height=image_height,
                        frames=frames,
                        fps=fps,
                        motion_type=motion_type,

                        # 地點標籤改到最終合檔時統一繪製。
                        watermark_filter="",
                    )
                )

                # 注意：
                # 此版本讀取單張圖片，
                # 由 zoompan 的 d=frames 產生前景影格。
                # 不要在輸入前加上 -loop 1，
                # 否則可能重複啟動整段運鏡。
                slide_cmd = [
                    ffmpeg,
                    "-y",
                    "-hide_banner",
                    "-loglevel", "error",

                    # 限制濾鏡並行度，降低多分支高解析度
                    # 處理時的瞬間資源需求。
                    "-filter_complex_threads", "1",

                    "-i", normalized_image,

                    "-filter_complex", filter_graph,
                    "-map", "[vout]",

                    "-frames:v", str(frames),
                    "-r", str(fps),
                    "-an",

                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "20",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",

                    slide_out,
                ]

                run_ffmpeg(
                    slide_cmd,
                    task_id,
                    (
                        f"第 {idx + 1} 張照片"
                        f"（{motion_names[motion_type]}）"
                    ),
                )

        # 3. 建立串接清單。
        with open(concat_list_path, "w", encoding="utf-8") as file:
            for slide in slide_videos:
                # concat 清單中的單引號需額外處理。
                slide_path = Path(slide).as_posix().replace(
                    "'", "'\\''"
                )
                file.write(f"file '{slide_path}'\n")

        # 4. 串接照片影片。
        concat_cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            temp_video_path,
        ]

        run_ffmpeg(concat_cmd, task_id, "串接幻燈片")

        # 5. 按字數分配字幕顯示時間。
        # 目前不讀取 vtt_file，保留參數以維持既有呼叫介面。
        # 只對整支影片做一次開場淡入。
        # 不對每張照片做淡出，避免切換時反覆變黑。
        opening_fade = min(0.3, audio_duration / 4)

        video_filter_arg = (
            "format=yuv420p,"
            f"fade=t=in:st=0:d={opening_fade:.6f}"
        )
        # 左上角地點標籤改在最終合檔時繪製，
        # 不參與照片運鏡與前景／背景合成。
        if not font_path:
            print(
                f"[任務 {task_id}] ⚠️ "
                "找不到中文字型，無法繪製左上角地點標籤"
            )
        else:
            label_filters = []
            frame_cursor = 0

            for idx, frame_count in enumerate(
                slide_frame_counts
            ):
                start_t = frame_cursor / fps
                frame_cursor += frame_count
                end_t = frame_cursor / fps

                meta_info = (
                    spot_metadata[idx]
                    if spot_metadata and idx < len(spot_metadata)
                    else {}
                )

                if not isinstance(meta_info, dict):
                    meta_info = {}

                spot_name = str(
                    meta_info.get("spot_name")
                    or merchant_name
                    or "探索據點"
                ).strip()

                codename = str(
                    meta_info.get("location_codename") or ""
                ).strip()

                visit_time = str(
                    meta_info.get("visit_time") or ""
                ).strip()

                # 分成兩行，避免直式影片的標籤太長。
                name_limit = (
                    18
                    if video_orientation == "portrait"
                    else 32
                )

                if len(spot_name) > name_limit:
                    spot_name = (
                        spot_name[:name_limit - 1] + "…"
                    )

                # 此處寫入純文字檔，
                # 不使用 clean_subtitle_text() 的濾鏡跳脫。
                first_line = f"地點：{spot_name}"

                details = []

                if codename:
                    details.append(
                        f"代號：{codename[:12]}"
                    )

                if visit_time:
                    details.append(
                        f"時間：{visit_time[:20]}"
                    )

                label_text = first_line

                if details:
                    label_text += "\n" + "  ".join(details)

                # 保留換行，其餘移除控制字元。
                label_text = "".join(
                    char
                    for char in label_text
                    if char == "\n" or char.isprintable()
                )

                # 使用隨機暫存檔名，避免 task_id 中特殊字元
                # 進入 FFmpeg 濾鏡路徑。
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    suffix=".txt",
                    prefix="vlog_label_",
                    dir=str(OUTPUT_DIR),
                    delete=False,
                ) as label_file:
                    label_file.write(label_text)
                    label_path = label_file.name

                label_text_paths.append(label_path)

                # Windows 磁碟代號的冒號需跳脫。
                # 單引號與濾鏡中的特殊字元也一併處理。
                escaped_label_path = (
                    Path(label_path)
                    .resolve()
                    .as_posix()
                    .replace(":", "\\:")
                    .replace("'", "'\\''")
                )

                # 起點包含、終點不包含，避免切換那一刻
                # 前後兩個地點標籤同時出現。
                label_filter = (
                    f"drawtext=fontfile='{font_path}'"
                    f":textfile='{escaped_label_path}'"
                    ":expansion=none"
                    ":fontcolor=white"
                    f":fontsize={profile['watermark_font_size']}"
                    ":line_spacing=8"
                    ":box=1"
                    ":boxcolor=black@0.55"
                    ":boxborderw=12"
                    ":shadowcolor=black@0.65"
                    ":shadowx=1"
                    ":shadowy=1"
                    ":x=40"
                    ":y=50"
                    f":enable='gte(t,{start_t:.6f})"
                    f"*lt(t,{end_t:.6f})'"
                )

                label_filters.append(label_filter)

                print(
                    f"[任務 {task_id}] 🏷️ "
                    f"地點標籤 {idx + 1}："
                    f"{start_t:.3f}s～{end_t:.3f}s，"
                    f"{label_text!r}"
                )

            if label_filters:
                video_filter_arg += (
                    "," + ",".join(label_filters)
                )

        if subtitle_text and font_path:
            sentences = split_subtitle_sentences(
                subtitle_text,
                max_len=profile["subtitle_max_len"],
            )

            total_chars = sum(len(s) for s in sentences)
            current_t = 0.0
            drawtexts = []

            if total_chars:
                for sentence in sentences:
                    duration = (
                        len(sentence) / total_chars
                    ) * audio_duration

                    end_t = current_t + duration
                    safe_sub = clean_subtitle_text(sentence)

                    if safe_sub:
                        drawtexts.append(
                            f"drawtext=fontfile='{font_path}'"
                            f":text='{safe_sub}'"
                            ":fontcolor=white"
                            f":fontsize={profile['subtitle_font_size']}"
                            ":box=1:boxcolor=black@0.6"
                            ":boxborderw=10"
                            ":x=(w-text_w)/2"
                            f":y=h-{profile['subtitle_bottom']}"
                            f":enable='between(t,{current_t:.3f},"
                            f"{end_t:.3f})'"
                        )

                    current_t = end_t

            if drawtexts:
                video_filter_arg += "," + ",".join(drawtexts)

        elif subtitle_text:
            print(
                f"[任務 {task_id}] ⚠️ 找不到中文字型，略過字幕"
            )

        # 6. 合併旁白與可選的背景音樂。
        has_bgm = bool(
            bgm_file and os.path.isfile(str(bgm_file))
        )

        final_cmd = [
            ffmpeg, "-y",
            "-i", temp_video_path,
            "-i", str(tts_audio_file),
        ]

        if has_bgm:
            final_cmd += [
                "-i", str(bgm_file),
                "-filter_complex",
                (
                    "[1:a]volume=1.0[a1];"
                    "[2:a]volume=0.2[a2];"
                    "[a1][a2]amix=inputs=2:duration=first[aout]"
                ),
            ]

        final_cmd += [
            "-vf", video_filter_arg,
            "-map", "0:v:0",
            "-map", "[aout]" if has_bgm else "1:a:0",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            building_output_path,
        ]

        print(
            f"[任務 {task_id}] 最終合檔指令：\n"
            f"{subprocess.list2cmdline(final_cmd)}"
        )

        run_ffmpeg(final_cmd, task_id, "最終合檔")

        # 7. 驗證輸出檔案與音軌。
        if (
            not os.path.isfile(building_output_path)
            or os.path.getsize(building_output_path) == 0
        ):
            raise RuntimeError(
                "ffmpeg 執行完畢但找不到有效輸出檔案"
            )

        probe_result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                building_output_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if (
            probe_result.returncode != 0
            or not probe_result.stdout.strip()
        ):
            raise RuntimeError(
                "合成影片缺少音軌，或無法讀取："
                + probe_result.stderr[-1000:]
            )

               # 8. 完成後才發布正式檔名。
        os.replace(
            building_output_path,
            final_output_path,
        )

        print(
            f"[任務 {task_id}] ✅ 影片合成成功："
            f"{final_output_path}"
        )

    except Exception as e:
        print(f"[任務 {task_id}] ❌ 合成崩潰：{e}")
        traceback.print_exc()
        raise

    finally:
        cleanup_paths = [
            concat_list_path,
            temp_video_path,
            building_output_path,
        ] + slide_videos + label_text_paths

        for path in cleanup_paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as cleanup_error:
                    print(
                        f"[任務 {task_id}] ⚠️ 無法清除暫存檔："
                        f"{path}，原因：{cleanup_error}"
                    )
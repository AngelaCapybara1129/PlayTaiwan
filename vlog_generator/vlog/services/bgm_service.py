import hashlib
from pathlib import Path

from core.config import BGM_LIBRARY_DIR, BGM_MOODS


SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
}


def _find_tracks(mood: str) -> list[Path]:
    """取得指定分類中可用的音樂，排除曲庫外部路徑。"""
    root = Path(BGM_LIBRARY_DIR).resolve()
    folder = root / mood

    if not folder.is_dir():
        return []

    tracks = []

    for candidate in folder.iterdir():
        if candidate.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue

        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)

            if resolved.is_file() and resolved.stat().st_size > 0:
                tracks.append(resolved)

        except (OSError, ValueError):
            continue

    return sorted(
        tracks,
        key=lambda path: str(path).casefold(),
    )


def select_bgm(
    mood: str,
    selection_key: str = "",
) -> dict:
    """
    從本機分類曲庫選曲。

    指定分類沒有音樂時，改用 neutral。
    neutral 也沒有可用音樂時，明確報錯。
    """
    if not isinstance(mood, str) or mood not in BGM_MOODS:
        raise ValueError(f"不支援的配樂分類：{mood!r}")

    requested_mood = mood
    selected_mood = mood
    tracks = _find_tracks(selected_mood)

    if not tracks and selected_mood != "neutral":
        selected_mood = "neutral"
        tracks = _find_tracks(selected_mood)

    if not tracks:
        raise FileNotFoundError(
            "找不到可用的背景音樂。"
            f"請在 {BGM_LIBRARY_DIR} 下放入分類音樂，"
            "並至少準備一首 neutral 配樂。"
        )

    digest = hashlib.sha256(
        f"{selection_key}:{selected_mood}".encode("utf-8")
    ).digest()

    index = int.from_bytes(digest[:8], "big") % len(tracks)
    selected_path = tracks[index]

    return {
        "path": str(selected_path),
        "track_id": f"{selected_mood}/{selected_path.name}",
        "file_name": selected_path.name,
        "requested_mood": requested_mood,
        "selected_mood": selected_mood,
        "used_fallback": selected_mood != requested_mood,
    }
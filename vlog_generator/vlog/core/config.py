from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR = BASE_DIR / "assets"
IMAGES_DIR = ASSETS_DIR / "images"
AUDIO_DIR = ASSETS_DIR / "audio"
BGM_DIR = ASSETS_DIR / "bgm"
OUTPUT_DIR = BASE_DIR / "output"

VLOG_WIDTH = 1080
VLOG_HEIGHT = 1920
FPS = 30

# 模板配置
TEMPLATES = {
    "user_vlog": {
        "name": "一般生活記錄",
        "image_duration": 2.5,
        "add_cta": False,
        "bgm_vol": 0.3,
        "tone": "輕鬆、口語、充滿個人旅遊日記感的情緒"
    },
    "merchant_promo": {
        "name": "商家宣傳行銷",
        "image_duration": 3.5,
        "add_cta": True,
        "bgm_vol": 0.4,
        "tone": "專業、熱情、具吸引力的商業口吻，強調整體質感與獨家優惠"
    }
}

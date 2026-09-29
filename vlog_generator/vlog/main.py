import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from api.routes import router as vlog_router
from services.llm_service import preload_llm

COMFY_URL = "http://127.0.0.1:8188"

# ComfyUI 資料夾：優先使用環境變數 COMFY_DIR，其次依序嘗試下列路徑
COMFY_DIR_CANDIDATES = [
    os.environ.get("COMFY_DIR", ""),
    r"D:\playtaiwan\ComfyUI",
    "/home/jackstar/playtaiwan/ComfyUI",
]


def _find_comfy_dir():
    for d in COMFY_DIR_CANDIDATES:
        if d and os.path.isdir(d):
            return Path(d)
    return None


def _find_comfy_python(comfy_dir: Path) -> str:
    """優先使用 ComfyUI 自己的虛擬環境，找不到就用目前的 Python"""
    for rel in (".venv/Scripts/python.exe", ".venv/bin/python", "venv/Scripts/python.exe", "venv/bin/python"):
        p = comfy_dir / rel
        if p.exists():
            return str(p)
    return sys.executable


def _start_comfy_if_needed():
    """檢查 ComfyUI 是否在跑，沒有就嘗試在背景啟動 (同步函式，會在執行緒中執行)"""
    try:
        response = requests.get(COMFY_URL, timeout=2)
        if response.status_code == 200:
            print("🎨 [ComfyUI] 偵測到 ComfyUI 已經在背景運行中！")
            return
    except requests.exceptions.RequestException:
        pass  # 連線失敗、逾時等，都視為尚未啟動

    print("⚠️ [ComfyUI] 尚未啟動，正在嘗試自動啟動 ComfyUI (Port 8188)...")
    comfy_dir = _find_comfy_dir()
    if not comfy_dir:
        print(f"❌ 找不到 ComfyUI 資料夾，已嘗試: {[d for d in COMFY_DIR_CANDIDATES if d]}")
        print("   可設定環境變數 COMFY_DIR 指向 ComfyUI 資料夾。")
        return

    try:
        subprocess.Popen(
            [_find_comfy_python(comfy_dir), "main.py"],
            cwd=str(comfy_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"🚀 [ComfyUI] 已於背景啟動 ({comfy_dir})")
    except Exception as e:
        print(f"❌ [ComfyUI] 啟動失敗: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1) ComfyUI 檢查 (放到執行緒，避免卡住事件迴圈)
    await asyncio.to_thread(_start_comfy_if_needed)

    # 2) 預載 LLM，避免第一個請求才載入模型而逾時 (可用 PRELOAD_LLM=0 關閉)
    if os.environ.get("PRELOAD_LLM", "1") != "0":
        try:
            await asyncio.to_thread(preload_llm)
        except Exception as e:
            print(f"❌ LLM 預載失敗 (之後第一次請求會再嘗試載入): {e}")

    yield


app = FastAPI(
    title="智慧觀光 Vlog 引擎 (低耦合架構)",
    docs_url="/api/docs",     # 設定 Swagger UI 的對應路徑
    redoc_url="/api/redoc",   # 設定 ReDoc 的對應路徑
    lifespan=lifespan,
)


# 強制修正 OpenAPI 規格，確保檔案上傳元件在 Swagger 中正確渲染，避免顯示亂碼
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="0.1.0",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# 將 api/routes.py 寫好的端點掛載上來
app.include_router(vlog_router, prefix="/api")

if __name__ == "__main__":
    # reload=True 每次存檔都會重啟並重新載入 8B 模型，且背景任務會中斷。
    # 開發時需要熱重載再設環境變數 UVICORN_RELOAD=1。
    use_reload = os.environ.get("UVICORN_RELOAD", "0") == "1"
    uvicorn.run("main:app", host="0.0.0.0", port=2026, reload=use_reload)
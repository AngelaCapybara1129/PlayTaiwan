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
from core.config import COMFY_DIR, COMFY_DIR_CANDIDATES, COMFY_URL
from services.llm_service import preload_llm


def _check_gpu():
    """啟動時檢查 GPU / PyTorch 是否相容，及早發現 sm_120 (RTX 50 系列) 不支援的問題"""
    try:
        import torch
    except Exception as e:
        print(f"❌ 無法匯入 torch: {e}")
        return

    if not torch.cuda.is_available():
        print("❌ [GPU] CUDA 不可用")
        return

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    print(f"🖥️ [GPU] {torch.cuda.get_device_name(0)} ({arch}), torch {torch.__version__}")
    if arch not in supported:
        print(f"❌ [GPU] 目前 PyTorch 不支援 {arch}，請升級到 cu128 以上。")
        print(f"   支援清單: {supported}")
        print(f"   Python: {sys.executable}")


def _find_comfy_python(comfy_dir: Path) -> str:
    """優先使用 ComfyUI 自己的虛擬環境，找不到就用目前的 Python"""
    for rel in (
        ".venv/Scripts/python.exe",
        ".venv/bin/python",
        "venv/Scripts/python.exe",
        "venv/bin/python",
    ):
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
    if not COMFY_DIR:
        print(f"❌ 找不到 ComfyUI 資料夾，已嘗試: {COMFY_DIR_CANDIDATES}")
        print("   可設定環境變數 COMFY_DIR 指向 ComfyUI 資料夾。")
        return

    try:
        # 把輸出寫進 log 檔，ComfyUI 啟動失敗時才看得到原因
        log_path = COMFY_DIR / "comfy_autostart.log"
        log_file = open(log_path, "a", encoding="utf-8")
        subprocess.Popen(
            [
                _find_comfy_python(COMFY_DIR),
                "main.py",
                "--listen", "0.0.0.0",
                "--port", "8188",
            ],
            cwd=str(COMFY_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        print(f"🚀 [ComfyUI] 已於背景啟動 ({COMFY_DIR})")
        print(f"   啟動紀錄: {log_path}")
    except Exception as e:
        print(f"❌ [ComfyUI] 啟動失敗: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 0) 檢查 GPU 與 PyTorch 是否相容
    _check_gpu()

    # 1) 先預載 LLM，讓它先取得 VRAM (可用 PRELOAD_LLM=0 關閉)
    if os.environ.get("PRELOAD_LLM", "1") != "0":
        try:
            await asyncio.to_thread(preload_llm)
        except Exception as e:
            print(f"❌ LLM 預載失敗 (之後第一次請求會再嘗試載入): {e}")

    # 2) LLM 載入完成後，再啟動 ComfyUI，使用剩下的 VRAM
    await asyncio.to_thread(_start_comfy_if_needed)

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
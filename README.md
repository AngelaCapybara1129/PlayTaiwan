# 🎬 PlayTaiwan AI Vlog & NPC 智慧生成系統

本專案結合 FastAPI、Whisper（ASR）、Wav2Vec2（SER）、Llama 3（LLM）、Neo4j（RAG 圖形資料庫）、Edge-TTS、XTTS Clone 與 ComfyUI，應用於觀光推廣與在地數位創新，提供 Vlog、NPC 互動語音及 AI 明信片生成。

目前 NPC 語音端點切換為 XTTS Clone 模式；Edge-TTS voice 代碼與 XTTS speaker 樣本之間的對應，以後端實際設定為準。

---

## 📂 專案結構

以下以目前 Windows 工作目錄為例：

```text
D:\playtaiwan\
├── ComfyUI/                       # 繪圖服務，Port 8188
├── xtts_server/                   # XTTS 語音服務，Port 8020
│   ├── .venv/                     # XTTS 獨立虛擬環境
│   ├── speakers/                  # NPC 參考音檔
│   ├── xtts_models/               # XTTS 模型
│   └── output/                    # XTTS 輸出
└── vlog_generator/
    ├── vlog/
    │   ├── .venv/                 # 主後端虛擬環境
    │   ├── assets/
    │   │   ├── images/            # 觀光照片與 ZIP 素材
    │   │   ├── audio/             # 語音與 ASR 音訊
    │   │   ├── bgm/               # 背景音樂
    │   │   └── dbs/               # Neo4j 備份，不納入 Git
    │   ├── output/                # 影片、語音與明信片
    │   ├── core/
    │   │   └── config.py
    │   ├── services/
    │   │   ├── asr_service.py
    │   │   ├── ser_service.py
    │   │   ├── llm_service.py
    │   │   ├── tts_service.py
    │   │   ├── neo4j_service.py
    │   │   ├── postcard_service.py
    │   │   └── moviepy_service.py
    │   ├── api/
    │   │   └── routes.py
    │   └── main.py
    ├── requirements.txt
    └── .gitignore
```

> 不同部署環境的虛擬環境位置可能不同，請依實際目錄調整。XTTS 使用模組啟動，本文件不使用不存在的 `app.py`。

## ⚙️ 主後端安裝與環境建置

### Windows（PowerShell）

```powershell
Set-Location "D:\playtaiwan\vlog_generator\vlog"
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 擇一安裝主後端依賴
uv pip install -r ../requirements.txt
# 或：
python -m pip install -r ../requirements.txt
```

### macOS / Linux

```bash
cd ~/playtaiwan/vlog_generator/vlog
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ../requirements.txt
```

> 三個服務各自使用自己的環境。以下啟動流程假設 ComfyUI 與 XTTS 的套件、模型及虛擬環境已完成建置，不會重新安裝或替換既有套件。

## 🚀 系統啟動方式（三服務架構）

| 服務 | 埠號 | 用途 |
|---|---|---|
| ComfyUI | 8188 | AI 明信片繪圖 |
| XTTS | 8020 | NPC 聲音複製與語音生成 |
| FastAPI 主後端 | 2026 | API、Vlog 任務與服務整合 |

請分別開啟三個終端機視窗並保持執行。以下 Windows 指令全部使用 PowerShell 語法。

### 步驟一：啟動 ComfyUI（PowerShell）

```powershell
Set-Location "D:\playtaiwan\ComfyUI"
.\.venv\Scripts\Activate.ps1
$env:CUDA_VISIBLE_DEVICES = "0"
python main.py --listen 0.0.0.0 --port 8188
```

若尚未安裝 ComfyUI 依賴，請在其環境中依專案 requirements 安裝，不要使用主後端或 XTTS 的環境。

### 步驟二：啟動 XTTS（PowerShell）

```powershell
Set-Location "D:\playtaiwan\xtts_server"

$env:Path = "D:\ffmpeg-master-latest-win64-gpl-shared\bin;" + $env:Path
$env:CUDA_LAUNCH_BLOCKING = "0"

.\.venv\Scripts\python.exe -m xtts_api_server -hs 127.0.0.1 -p 8020 -sf "D:\playtaiwan\xtts_server\speakers" -mf "D:\playtaiwan\xtts_server\xtts_models" -o "D:\playtaiwan\xtts_server\output"
```

服務位址：`http://127.0.0.1:8020`。

主後端的 XTTS 服務位址設定需與啟動時的 host、port 一致。此處只監聽本機；其他機器或容器的連線方式需另行配置。

#### NPC 參考音檔

先前 NPC 呼叫使用的 speaker 為 `阿達力.mp3`，請確認檔案實際位於：

```text
D:\playtaiwan\xtts_server\speakers\阿達力.mp3
```

查看檔案：

```powershell
Get-ChildItem "D:\playtaiwan\xtts_server\speakers"
```

`voice=zh-TW-YunJheNeural` 是原有 API 的輸入代碼；在 XTTS Clone 模式中，需要後端將它映射至有效 speaker。不要在未確認 routes/service 實作前，直接把 API voice 改成檔名。

### 步驟三：啟動 FastAPI 主後端（PowerShell）

```powershell
Set-Location "D:\playtaiwan\vlog_generator\vlog"
uv run python main.py
```

Swagger API 文件：`http://localhost:2026/api/docs`。

### macOS / Linux：原有服務啟動方式

ComfyUI：

```bash
cd ~/playtaiwan/ComfyUI
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python main.py --listen 0.0.0.0 --port 8188
```

主後端：

```bash
cd ~/playtaiwan/vlog_generator/vlog
uv run python main.py
```

XTTS 與 FFmpeg 的路徑需依 macOS / Linux 的實際安裝位置另行調整；不要直接使用本文件的 Windows 路徑。

## 📡 主要 API 功能與測試範例

### 1. AI 明信片：`POST /api/postcard/create_ai`

結合 Neo4j 知識圖譜、LLM 導覽文字及 ComfyUI，生成展覽圖錄風格明信片。

### 2. Vlog：`POST /api/visitor/create_vlog`

上傳語音、照片 ZIP、BGM 與景點清單，生成帶字幕的 Vlog。

以下為 Bash / macOS / Linux cURL 範例，請在 `vlog` 目錄執行：

```bash
curl -X POST "http://localhost:2026/api/visitor/create_vlog" \
  -H "accept: application/json" \
  -F "user_audio=@assets/audio/3.AI不是口號.m4a" \
  -F 'spot_list=["南投環湖茶園", "竹林秘境"]' \
  -F "image_zip=@assets/images/images.zip" \
  -F "bgm_file=@assets/bgm/ikoliks_aj-background-music-320427.mp3"
```

Windows 使用者可先透過 Swagger 上傳檔案測試，避免混用 Bash 的換行語法。

### 3. NPC 配音：`POST /api/npc/speak`

目前使用 XTTS Clone 模式，接收 `application/x-www-form-urlencoded`：

| 欄位 | 範例 | 說明 |
|---|---|---|
| text | 年輕人，這條街的故事我最熟了！ | NPC 台詞 |
| voice | zh-TW-YunJheNeural | 對應 NPC 的 voice 代碼 |
| rate | +15% | 原 API 的語速欄位 |
| pitch | +10Hz | 原 API 的音高欄位 |

> XTTS 模式是否實際套用 rate / pitch，及輸出音訊格式，需以後端實作為準；保留參數不代表已支援相同的 Edge-TTS 控制效果。

### 4. 任務狀態與下載

- `GET /api/check_status/{task_id}`：查詢背景影音任務進度。
- `GET /api/download/{filename}`：下載產出檔案。

## 🛠️ 常見問題

### PowerShell 無法辨識 `%PATH%` 或 `cd /d`

這些是 CMD 語法。PowerShell 使用：

```powershell
Set-Location "D:\playtaiwan\xtts_server"
$env:Path = "D:\ffmpeg-master-latest-win64-gpl-shared\bin;" + $env:Path
$env:CUDA_LAUNCH_BLOCKING = "0"
```

### `python app.py`：找不到檔案

目前提供的 XTTS 資料夾沒有 `app.py`。請使用前述 `python -m xtts_api_server` 啟動流程。

### `No module named xtts_api_server`

先確認套件是否存在於指定的虛擬環境：

```powershell
.\.venv\Scripts\python.exe -m pip show xtts-api-server
```

若同時顯示 `No module named pip`，先處理該環境的套件管理方式，不要改用系統 pip 盲目安裝。

### 啟動參數不被接受

不同版本可能提供不同選項，請查看目前版本的說明：

```powershell
.\.venv\Scripts\python.exe -m xtts_api_server --help
```

### `Speaker 阿達力.mp3 not found`

確認 speakers 內有該音檔，且後端 speaker 映射與實際檔名一致。XTTS 啟動時明確指定 `-sf`，可避免使用錯誤的相對目錄。

### FFmpeg 檢查

在已設定 Path 的終端機中執行：

```powershell
ffmpeg -version
```

保留實際 shared 版本的完整資料夾內容。若仍有音訊解碼錯誤，需進一步檢查缺失 DLL 與套件相容性，不能只以 `ffmpeg -version` 成功認定所有相依已正常。

### CUDA 環境

F5-TTS 與 XTTS 的虛擬環境不同。先前在 F5-TTS 安裝的 PyTorch 不會自動更新 XTTS 的環境。

`CUDA_LAUNCH_BLOCKING=0` 是目前採用的設定，不是顯卡編號，也不能用來修復缺少 GPU kernel 的問題。

## 🙀 服務埠號被佔用（macOS / Linux）

優先在原服務視窗按 Ctrl+C 正常停止。若仍有背景行程，以 Port 8188 為例：

```bash
sudo lsof -i :8188
# 若未安裝 lsof，可使用：
sudo ss -lptn 'sport = :8188'
```

確認 PID 與程式確實是要停止的服務，再終止：

```bash
sudo kill 12345
# 僅在無法正常結束時使用：
sudo kill -9 12345
```

將 `12345` 替換成實際 PID。此段不適用於 Windows PowerShell。XTTS 與主後端的埠號分別為 8020、2026。

## 🛑 Git 忽略規範

原專案設定預期排除：

- `.venv/`：虛擬環境。
- `output/`：生成影音與圖片。
- `assets/dbs/*.dump`：大型 Neo4j 備份。
- `__pycache__/`：Python 暫存。

新增 XTTS 後，請另外確認 `xtts_models/` 與 speakers 參考音檔的版控規則，避免提交大型模型或未獲授權的聲音樣本。是否已忽略，需檢查實際 `.gitignore`。

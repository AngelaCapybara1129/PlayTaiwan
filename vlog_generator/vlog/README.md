# 🎬 PlayTaiwan AI Vlog & NPC 智慧生成系統

本專案是一個基於 **FastAPI**、**Whisper (ASR)**、**Wav2Vec2 (SER)**、**Llama 3.2 (LLM)**、**Neo4j (RAG 圖形資料庫)**、**XTTSv2 (Zero-shot 語音複製)** 與 **ComfyUI (RTX 5090 繪圖引擎)** 的智慧影音與 AI 明信片生成系統。專為觀光推廣與在地數位創新設計，能自動分析語音情緒與內容，結合圖形資料庫脈絡，產出帶有精美自動配音的高質感 Vlog 影片、完美還原角色聲線的 NPC 互動語音，以及展覽圖錄風格的 AI 明信片。

---

## 📂 專案結構 (微服務架構)

```text
playtaiwan/
├── vlog_generator/vlog/    # 🚀 主程式 (FastAPI 後端)
│   ├── assets/             # 素材資料夾
│   │   ├── images/         # 存放靜態觀光照片 (ZIP 壓縮檔上傳解析)
│   │   ├── audio/          # 存放語音檔與 ASR 轉換音訊
│   │   ├── bgm/            # 存放背景音樂 (mp3)
│   │   └── dbs/            # (Git 略過) Neo4j 資料庫備份與檔案
│   ├── output/             # 影片、語音與 AI 明信片輸出資料夾
│   ├── core/               # 核心設定層
│   ├── services/           # 業務邏輯層
│   │   ├── asr_service.py  # 語音辨識服務
│   │   ├── ser_service.py  # 語音情感分析服務
│   │   ├── llm_service.py  # LLM 腳本與提示詞生成 (Llama 3.2)
│   │   ├── tts_service.py  # 呼叫 XTTS 微服務進行在地化語音複製
│   │   ├── neo4j_service.py# Neo4j 知識圖譜 RAG 檢索
│   │   ├── postcard_service.py # 呼叫 ComfyUI 生成 AI 明信片
│   │   └── moviepy_service.py # FFmpeg 影片串接與字幕燒錄
│   ├── api/                # 路由層 (包含 routes.py)
│   └── main.py             # 程式進入點 (Port 2026)
│
├── xtts_server/            # 🗣️ XTTS 語音生成獨立微服務
│   └── .venv/              # 獨立的虛擬環境 (避免套件衝突)
│
├── npc_audio/              # 🎙️ NPC 語音 Clone 參考檔 (.wav)
│
├── ComfyUI/                # 🎨 ComfyUI 繪圖引擎
└── requirements.txt        # 依賴套件清單

```

---

## ⚙️ 快速安裝與環境建置

### 1. 系統必備元件 (FFmpeg)

本系統高度依賴 FFmpeg 進行影音合成與音訊處理。請開啟全新的 PowerShell 終端機安裝：

```powershell
winget install ffmpeg

```

*(安裝完成後，請務必重新啟動所有終端機與編輯器以載入環境變數)*

### 2. 建立並啟動主程式虛擬環境 (推薦使用 `uv`)

**Windows（PowerShell）**

```powershell
cd vlog_generator/vlog
uv venv
.\.venv\Scripts\activate

```

### 3. 安裝依賴套件

透過高效能的 `uv` 安裝所有必要套件：

```powershell
uv pip install -r ../requirements.txt

```

---

## 🚀 系統啟動方式 (三軌並行微服務架構)

本系統將算力與功能分離，確保套件不衝突且能發揮最大效能。請開啟三個不同的終端機視窗，依序啟動以下服務：

### 步驟一：啟動 ComfyUI 繪圖引擎 (Port 8188)

切換至 ComfyUI 資料夾，啟動對應虛擬環境，並指定 GPU 啟動：

```powershell
cd D:\playtaiwan\ComfyUI
.\.venv\Scripts\activate
$env:CUDA_VISIBLE_DEVICES="0"; python main.py --listen 0.0.0.0 --port 8188

```

### 步驟二：啟動 XTTS 語音複製微服務 (Port 8020)

開啟第二個終端機，啟動 XTTS 專屬環境，並**掛載 NPC 參考音檔資料夾**：

```powershell
cd D:\playtaiwan\xtts_server
.\.venv\Scripts\activate
python -m xtts_api_server -sf "D:\playtaiwan\npc_audio"

```

*(啟動成功後將看到 `Latents created for all X speakers.`)*

### 步驟三：啟動 FastAPI 主程式服務 (Port 2026)

開啟第三個終端機，啟動 Vlog 引擎主後端：

```powershell
cd D:\playtaiwan\vlog_generator\vlog
.\.venv\Scripts\activate
uv run python main.py

```

🎉 服務全數啟動後，請前往 **Swagger API 互動文件** 進行測試：
👉 `http://localhost:2026/api/docs`

---

## 📡 主要 API 功能與測試範例

### 1. 🎨 AI 展覽圖錄風格明信片與導覽文字 (`POST /api/postcard/create_ai`)

結合 Neo4j 知識圖譜、Llama 在地短文生成與 RTX 5090 繪圖引擎，產出高質感彩色水粉插畫明信片。

### 2. 🎬 遊客端 Vlog 核心生成 (`POST /api/visitor/create_vlog`)

上傳語音檔、照片壓縮檔（ZIP）、BGM 與景點清單，自動產出帶有字幕的精美 Vlog 影片。

### 3. 🗣️ NPC 專屬配音 API (`POST /api/npc/speak`)

整合 XTTSv2 零樣本語音複製技術。前端只需傳入台詞與 NPC 名稱，系統將自動提取對應的 `.wav` 聲紋特徵，動態生成完美還原角色語氣的道地配音。

* **參數**：`text` (台詞)、`npc_name` (下拉選單直接選擇角色，如：珍奶奶、阿達力、薯光)。

### 4. 🔍 任務狀態與下載端點

* **`GET /api/check_status/{task_id}`**：查詢背景非同步影音任務進度。
* **`GET /api/download/{filename}`**：下載產出的影片、語音或明信片檔案。

---

## 🛑 開發注意事項 (.gitignore 規範)

為了避免上傳過大的檔案或破壞版控，以下項目已被 `.gitignore` 自動忽略：

* `.venv/` （虛擬環境）
* `output/` （生成的影音與圖片檔案）
* `assets/dbs/*.dump` （超過 100MB 的 Neo4j 資料庫備份）
* `__pycache__/` 與系統暫存檔

---

## 🙀 服務佔用排除 (Port 衝突)

當 Port 8188、8020 或 2026 被佔用時，通常是因為之前啟動的 Python 行程沒有被正確關閉，仍在背景默默運作。你可以透過以下指令強制釋放：

### Windows 環境 (PowerShell)

以 Port 8188 為例，直接執行以下一行指令，系統會找出佔用該 Port 的程式並強制關閉：

```powershell
Get-Process -Id (Get-NetTCPConnection -LocalPort 8188).OwningProcess | Stop-Process -Force

```

*(若要關閉 XTTS 或主程式，只需將指令中的 `8188` 替換為 `8020` 或 `2026` 即可)*

### Linux / macOS 環境

```bash
# 找出佔用的 PID
sudo lsof -i :8188
# 強制終止
sudo kill -9 <PID>

```

# 🎬 PlayTaiwan AI Vlog & NPC 智慧生成系統

本專案是一個基於 **FastAPI**、**Whisper (ASR)**、**Wav2Vec2 (SER)**、**Llama 3 (LLM)**、**Neo4j (RAG 圖形資料庫)**、**Edge-TTS** 與 **ComfyUI (RTX 5090 繪圖引擎)** 的智慧影音與 AI 明信片生成系統。專為觀光推廣與在地數位創新設計，能自動分析語音情緒與內容，結合圖形資料庫脈絡，產出帶有精美自動配音的高質感 Vlog 影片、NPC 互動語音以及展覽圖錄風格的 AI 明信片。

---

## 📂 專案結構

```text
vlog_generator/
├── vlog/
│   ├── assets/             # 素材資料夾
│   │   ├── images/         # 存放靜態觀光照片 (ZIP 壓縮檔上傳解析)
│   │   ├── audio/          # 存放語音檔與 ASR 轉換音訊
│   │   ├── bgm/            # 存放背景音樂 (mp3)
│   │   └── dbs/            # (Git 略過) Neo4j 資料庫備份與檔案
│   ├── output/             # 影片、語音與 AI 明信片輸出資料夾
│   ├── core/               # 核心設定層
│   │   └── config.py       # 全域路徑與常數設定
│   ├── services/           # 業務邏輯層
│   │   ├── asr_service.py  # 語音辨識服務
│   │   ├── ser_service.py  # 語音情感分析服務
│   │   ├── llm_service.py  # LLM 腳本與提示詞生成
│   │   ├── tts_service.py  # 台灣腔高音質旁白生成
│   │   ├── neo4j_service.py# Neo4j 知識圖譜 RAG 檢索
│   │   ├── postcard_service.py # ComfyUI AI 明信片生成服務
│   │   └── moviepy_service.py # FFmpeg 影片串接與字幕燒錄
│   ├── api/                # 路由層
│   │   └── routes.py       # FastAPI 端點 (/visitor/create_vlog, /postcard/create_ai 等)
│   └── main.py             # 程式進入點 (啟動 FastAPI 伺服器)
├── requirements.txt        # 依賴套件清單
└── .gitignore              # Git 忽略清單 (自動排除 .venv、output 等)

```

---

## ⚙️ 快速安裝與環境建置

### 1. 建立並啟動虛擬環境 (推薦使用 `uv`)

#### **Windows（PowerShell）**

```powershell
cd vlog_generator/vlog
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```

#### **macOS / Linux**

```bash
cd vlog_generator/vlog
python3 -m venv .venv
source .venv/bin/activate

```

### 2. 安裝依賴套件

透過高效能的 `uv` 或 `pip` 安裝所有必要套件：

```bash
uv pip install -r ../requirements.txt
# 或使用標準 pip
pip install -r ../requirements.txt

```

---

## 🚀 系統啟動方式 (雙軌並行架構)

本系統結合了 **ComfyUI 繪圖引擎 (Port 8188)** 與 **FastAPI 後端引擎 (Port 2026)**，請依序啟動以下服務：

### 步驟一：啟動 ComfyUI 繪圖引擎 (指定顯卡與外部監聽)

請切換至 ComfyUI 資料夾、啟動對應虛擬環境（並確保安裝好 `sqlalchemy` 等相依套件），再指定 GPU 與 Port 8188 啟動：

```bash
cd ~/playtaiwan/ComfyUI
source .venv/bin/activate
# 若尚未安裝相依套件，請先執行：pip install -r requirements.txt
$env:CUDA_VISIBLE_DEVICES="0"; python main.py --listen 0.0.0.0 --port 8188

```

### 步驟二：啟動 FastAPI 主程式服務

開啟另一個終端機視窗，啟動你的 Vlog 引擎主後端（運行於 Port 2026）：

```bash
cd ~/playtaiwan/vlog_generator/vlog
uv run python main.py

```

服務啟動後，你可以前往 **Swagger API 互動文件** 進行測試：
👉 `http://localhost:2026/api/docs`

---

## 📡 主要 API 功能與測試範例

### 1. 🎨 AI 展覽圖錄風格明信片與導覽文字 (`POST /api/postcard/create_ai`)

結合 Neo4j 知識圖譜、Llama 在地短文生成與 RTX 5090 繪圖引擎，產出高質感彩色水粉插畫明信片。

### 2. 🎬 遊客端 Vlog 核心生成 (`POST /api/visitor/create_vlog`)

上傳語音檔、照片壓縮檔（ZIP）、BGM 與景點清單，自動產出帶有字幕的精美 Vlog 影片。

**API 測試範例指令 (cURL)：**

```bash
curl -X POST "http://localhost:2026/api/visitor/create_vlog" \
  -H "accept: application/json" \
  -F "user_audio=@assets/audio/3.AI不是口號.m4a" \
  -F "spot_list=[\"南投環湖茶園\", \"竹林秘境\"]" \
  -F "image_zip=@assets/images/images.zip" \
  -F "bgm_file=@assets/bgm/ikoliks_aj-background-music-320427.mp3"

```

### 3. 🗣️ NPC 專屬配音 API (`POST /api/npc/speak`)

輸入文字，動態生成活潑可愛的 NPC 專屬配音（mp3）。

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

## 🙀服務佔用
當 Port 8188（通常是你的 ComfyUI 繪圖引擎）被佔用時，通常是因為之前啟動的 Python 行程沒有被正確關閉，仍在背景默默運作。

你可以透過以下步驟找出並終止佔用該 Port 的行程：

### 步驟一：找出是哪個行程佔用了 Port 8188

在終端機輸入以下指令：

```bash
sudo lsof -i :8188

```

*(如果系統提示 `lsof: command not found`，可以改用 `sudo ss -lptn 'sport = :8188'`)*

執行後，你會看到類似下面的輸出：

```text
COMMAND   PID USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
python3  12345 jackstar    3u  IPv4  ...      0t0 TCP *:8188 (LISTEN)

```

注意看 **`PID`**（例如上面的 `12345`）。

---

### 步驟二：強制終止該行程

拿到 PID 之後，使用 `kill` 指令將它關閉：

```bash
sudo kill -9 12345

```

*(請將 `12345` 換成你畫面中實際看到的數字)*

---

### 步驟三：重新啟動服務

行程被殺掉後，Port 就釋放了。這時你就可以再次順利啟動你的 ComfyUI：

```bash
cd ~/playtaiwan/ComfyUI
source .venv/bin/activate
$env:CUDA_VISIBLE_DEVICES="0"; python main.py --listen 0.0.0.0 --port 8188

```
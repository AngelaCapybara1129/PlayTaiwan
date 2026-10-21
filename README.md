# 🎬 PlayTaiwan AI Vlog & NPC 智慧生成系統

PlayTaiwan 整合 FastAPI、語音辨識、情緒辨識、生成式語言模型、Neo4j、XTTS 與 ComfyUI，提供旅遊故事、互動任務、導覽角色語音、旅遊紀錄影片及 AI 明信片生成。

目前主後端實測使用：

- 語言模型：`unsloth/Llama-3.2-3B-Instruct`
- 模型精度：`bfloat16`
- 文字向量模型：`intfloat/multilingual-e5-large`
- 向量維度：1024
- 圖形資料庫：Neo4j
- 導覽角色語音：目前採 XTTS Clone，實際 speaker 對應由後端設定決定
- 圖像生成：ComfyUI
- API 與非同步工作：FastAPI、專案內建 `job_queue`

其他整合模組包含 Whisper（ASR）、Wav2Vec2（SER）與 MoviePy。實際模型、服務位址及路徑以專案設定為準。

> 本專案的生成結果需要通過結構與文字檢查，但這些檢查不等於完整的事實查核、謎題可解性證明或現場營業驗證。

---

## 📌 目前功能狀態

| 功能 | 目前狀態 |
|---|---|
| 自然語言旅遊需求解析 | 已接入 |
| Neo4j 地點資料讀取與 E5 語意候選檢索 | 已接入 |
| 候選需求審核與原文引用比對 | 已接入，仍需持續評估誤拒與誤放 |
| 單套故事、多節點與每站多任務 | 已接入 |
| 每套故事選定一位導覽角色 | 已接入 |
| 角色介紹、每站開場與過關台詞 | 已接入 |
| 需求不清楚時提供引導 | 已接入，尚非完整意圖辨識 |
| 沒有符合候選時回傳 `no_match` | 已接入 |
| 已知生成耗盡與服務問題的結果分流 | 已接入，需確認各拋錯位置使用專用例外 |
| 多樣化、已驗證的備援劇本庫 | 規劃中，尚未啟用 |
| 營業時間資料讀取並提供給候選審核 | 已接入 |
| 指定日期、星期與跨午夜的確定性開放判斷 | 尚未完成 |
| 交通路線、移動時間與可達性最佳化 | 尚未完成 |
| Vlog 草稿預覽與正式產片 | 已提供 API |
| NPC 語音生成 | 已提供排隊 API |
| AI 明信片生成 | 已提供 API |

---

## 📂 專案結構

以下以目前 Windows 工作目錄為例。結構列出主要模組，其他檔案以實際專案為準。

```text
D:\playtaiwan\
├── ComfyUI/                         # 圖像生成服務，Port 8188
├── xtts_server/                     # XTTS 語音服務，Port 8020
│   ├── .venv/                       # XTTS 獨立虛擬環境
│   ├── speakers/                    # 經授權的角色參考音檔
│   ├── xtts_models/                 # XTTS 模型
│   └── output/                      # XTTS 輸出
└── vlog_generator/
    ├── vlog/
    │   ├── .venv/                   # 主後端虛擬環境
    │   ├── .env                     # 本機設定與憑證，不提交 Git
    │   ├── assets/
    │   │   ├── images/
    │   │   ├── audio/
    │   │   ├── bgm/
    │   │   └── dbs/                 # Neo4j 備份，不提交 Git
    │   ├── output/
    │   ├── core/
    │   │   └── config.py
    │   ├── api/
    │   │   ├── routes.py
    │   │   ├── schemas.py
    │   │   ├── story_task_schemas.py
    │   │   └── story_generation_schemas.py
    │   ├── services/
    │   │   ├── asr_service.py
    │   │   ├── ser_service.py
    │   │   ├── llm_service.py
    │   │   ├── neo4j_rag_service.py
    │   │   ├── blueprint_auto_service.py
    │   │   ├── story_workflow_service.py
    │   │   ├── story_workflow_errors.py
    │   │   ├── story_tasks_service.py
    │   │   ├── story_text_fields.py
    │   │   ├── node_text_generator.py
    │   │   ├── task_json_generator.py
    │   │   ├── story_text_fields.py
    │   │   ├── zh_tw_guard.py
    │   │   ├── job_queue.py
    │   │   ├── tts_service.py
    │   │   ├── bgm_service.py
    │   │   ├── postcard_service.py
    │   │   └── moviepy_service.py
    │   └── main.py
    ├── requirements.txt
    └── .gitignore
```

> 不同部署環境的虛擬環境位置可能不同，請依實際目錄調整。XTTS 使用模組啟動，不使用 `python app.py`。

---

## ⚙️ 主後端安裝與環境設定

### Windows PowerShell

```powershell
Set-Location "D:\playtaiwan\vlog_generator\vlog"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 擇一安裝：
uv pip install -r ..\requirements.txt

# 或：
python -m pip install -r ..\requirements.txt
```

### macOS / Linux

```bash
cd ~/playtaiwan/vlog_generator/vlog

python3 -m venv .venv
source .venv/bin/activate

python -m pip install -r ../requirements.txt
```

### 使用既有環境的注意事項

- 主後端、ComfyUI 與 XTTS 各自使用獨立環境。
- 已可正常執行的環境，不應只為消除警告而升級或重裝全部套件。
- PyTorch、CUDA、顯示卡與各 AI 模組的相容性，需依實際環境確認。
- 不要把另一個服務環境中的套件，當成主後端已安裝的套件。

### `.env` 範例

在 `vlog` 目錄準備 `.env`：

```dotenv
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=請填入本機密碼
NEO4J_DATABASE=playtaiwandb
NEO4J_QUERY_TIMEOUT=30

LLM_MODEL=unsloth/Llama-3.2-3B-Instruct
LLM_QUANT=none
LLM_BATCH_SIZE=1
LLM_OPENCC=1

EMBEDDING_DEVICE=cpu
NEO4J_ALLOW_RAW_READ_QUERY=0
```

說明：

- `LLM_BATCH_SIZE=1` 是保守設定範例，實際批次大小依效能測試調整。
- `EMBEDDING_DEVICE=cpu` 表示文字向量模型使用 CPU。
- `NEO4J_ALLOW_RAW_READ_QUERY=0` 表示停用自訂 Cypher。
- Neo4j 設定必須在匯入依賴這些設定的專案模組前載入。
- XTTS、ComfyUI 與輸出路徑請依 `core/config.py` 和相關 service 實作設定，不假設本文件未列出的環境變數已被支援。
- 不要提交真實密碼、Token 或 `.env`。

---

## 🚀 系統啟動方式

主應用整合三個常駐服務，另需可連線的 Neo4j。

| 服務 | 埠號 | 用途 |
|---|---|---|
| ComfyUI | 8188 | AI 明信片圖像生成 |
| XTTS | 8020 | 角色聲音複製與語音生成 |
| FastAPI 主後端 | 2026 | API、工作佇列與服務整合 |
| Neo4j Bolt | 依設定，範例為 7687 | 地點與知識圖譜查詢 |

請分別開啟服務終端機並保持執行。

### 步驟一：啟動 ComfyUI

```powershell
Set-Location "D:\playtaiwan\ComfyUI"

.\.venv\Scripts\Activate.ps1

$env:CUDA_VISIBLE_DEVICES = "0"

python main.py --listen 0.0.0.0 --port 8188
```

請使用 ComfyUI 自己的套件與虛擬環境。

### 步驟二：啟動 XTTS

```powershell
Set-Location "D:\playtaiwan\xtts_server"

$env:Path = "D:\ffmpeg-master-latest-win64-gpl-shared\bin;" + $env:Path
$env:CUDA_LAUNCH_BLOCKING = "0"

.\.venv\Scripts\python.exe -m xtts_api_server `
  -hs 127.0.0.1 `
  -p 8020 `
  -sf "D:\playtaiwan\xtts_server\speakers" `
  -mf "D:\playtaiwan\xtts_server\xtts_models" `
  -o "D:\playtaiwan\xtts_server\output"
```

服務位址：

```text
http://127.0.0.1:8020
```

主後端設定需與 XTTS 的 host、port 一致。上述設定只監聽本機，容器或其他機器的連線需另外配置。

確認參考音檔：

```powershell
Get-ChildItem "D:\playtaiwan\xtts_server\speakers"
```

參考音檔應使用已取得適當授權的聲音素材。角色名稱與 speaker 檔名的映射，以 `tts_service.py` 為準。

### 步驟三：啟動 FastAPI 主後端

```powershell
Set-Location "D:\playtaiwan\vlog_generator\vlog"

$env:TRANSFORMERS_VERBOSITY = "error"

uv run python main.py
```

也可明確指定既有虛擬環境：

```powershell
.\.venv\Scripts\python.exe main.py
```

常用位址：

```text
Swagger：http://localhost:2026/api/docs
OpenAPI：http://localhost:2026/openapi.json
```

`TRANSFORMERS_VERBOSITY=error` 只降低 Transformers 的日誌量，不會修復參數相容性，也不會隱藏所有 Torch 或專案日誌。

### macOS / Linux 啟動範例

ComfyUI：

```bash
cd ~/playtaiwan/ComfyUI
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python main.py --listen 0.0.0.0 --port 8188
```

主後端：

```bash
cd ~/playtaiwan/vlog_generator/vlog

TRANSFORMERS_VERBOSITY=error uv run python main.py
```

XTTS、FFmpeg 與模型路徑請依實際安裝位置調整。Windows GPU 環境的成功測試，不代表其他平台已完成相同驗證。

---

## 🧭 自然語言故事 v2

### 處理原則

```text
正常需求照常生成
需求不清楚就引導
找不到就提出替代入口
生成失敗則回覆可理解的結果
有相容的已驗證劇本庫後，才啟用備援切換
```

備案不得假裝已完成原需求，也不能自行改掉縣市、指定對象或排除條件。

### 目前流程

```text
使用者輸入
    ↓
明確預設值與簡單模糊輸入檢查
    ↓
需求解析
    ↓
可搜尋條件檢查
    ↓
Neo4j / E5 候選檢索
    ↓
基本資料、狀態與行政區篩選
    ↓
模型需求審核與原文引用檢查
    ↓
選擇單一行政區的可用站點
    ↓
故事、角色介紹、章節與任務生成
    ↓
角色台詞生成
    ↓
劇本與業務結果驗證
```

### 提交需求

目前部署的實際路徑：

```text
POST /api/api/admin/generate_script_blueprint_by_text
```

> 此路徑目前包含兩層 `/api`，是現有路由與主應用前綴組合的結果。呼叫時請使用 Swagger 顯示的實際 URL，不要自行刪除其中一層。

Request body：

```json
{
  "user_prompt": "我們想要兩個人去看雲海"
}
```

目前 `user_prompt` 上限為 3000 字元。請求 Schema 原本設有預設需求；測試真實輸入時請明確填入，不要依賴預設值。

PowerShell 測試：

```powershell
$body = @{
    user_prompt = "我們想要兩個人去看雲海"
} | ConvertTo-Json

$job = Invoke-RestMethod `
    -Method Post `
    -Uri "http://localhost:2026/api/api/admin/generate_script_blueprint_by_text" `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))

$job
```

POST 會先回傳 HTTP 202 與任務資料，不代表劇本已完成。

回應包含：

```text
task_id
status
check_url
job_id
pipeline_version = auto-story-v2
```

### 查詢工作

```text
GET /api/jobs/{task_id}
```

PowerShell：

```powershell
Invoke-RestMethod `
    -Method Get `
    -Uri ("http://localhost:2026" + $job.check_url)
```

相容舊路由：

```text
GET /api/api/admin/generate_script_blueprint_by_text/{job_id}
```

優先使用 POST 回傳的 `check_url`。

---

## 📦 v2 結果格式

佇列執行狀態與業務結果分開：

```text
status
→ 佇列工作是否已完成

result.outcome
→ 這次工作實際提供什麼結果
```

因此：

```text
status = done
```

不一定代表已生成劇本，也可能是已完成需求引導或無結果回覆。

### 業務結果

| `outcome` | 意義 | 是否包含劇本 |
|---|---|---|
| `generated` | 客製化劇本生成成功 | 是 |
| `needs_clarification` | 需要補充地區、活動或其他需求 | 否 |
| `no_match` | 本次候選沒有通過可用性或需求審核 | 否 |
| `fallback` | 切換至相容的已驗證劇本 | 格式已預留，功能尚未啟用 |
| `temporarily_unavailable` | 生成耗盡、已知服務不可用或未預期問題 | 否 |

成功時的劇本位置：

```text
result.story_payload.story
result.story_payload.npc
result.story_payload.nodes
```

不是舊版的：

```text
result.story
result.npc
result.nodes
```

### 引導結果範例

```json
{
  "outcome": "needs_clarification",
  "message": "目前還無法確認你想去哪裡或體驗什麼，請補充地區或活動。",
  "reason_code": "needs_travel_preferences",
  "story_payload": null,
  "alternatives": [
    {
      "action": "submit_prompt",
      "label": "探索自然景觀",
      "user_prompt": "我們想探索自然景觀"
    },
    {
      "action": "edit_request",
      "label": "補充地區或想體驗的內容"
    }
  ]
}
```

### 替代操作

| `action` | 前端行為 |
|---|---|
| `submit_prompt` | 使用選項中的 `user_prompt` 提交新工作 |
| `edit_request` | 回到輸入介面，讓使用者修改，不自動提交 |
| `retry` | 使用者確認後重新提交原需求 |

重新提交應取得新任務 ID，不覆寫舊任務結果。

無結果回覆中的主題按鈕是重新搜尋入口，不代表系統已找到對應景點。

### 前端分流範例

```javascript
function handleStoryJob(job) {
  if (job.status !== "done" || !job.result) {
    return;
  }

  const result = job.result;

  if (
    result.outcome === "generated" ||
    result.outcome === "fallback"
  ) {
    const payload = result.story_payload;

    // 使用 payload.story、payload.npc、payload.nodes。
    return;
  }

  // 顯示 result.message 與 result.alternatives。
  // 不讀取 nodes，不直接開啟遊戲。
}
```

前端另需處理佇列執行中、工作失敗及查詢錯誤等情況。

---

## 👥 故事角色與任務

### 故事角色池

目前故事生成使用六位導覽角色：

- 薯光
- 珍奶奶
- 阿達力
- 墨先生
- 霓霓
- 阿吉伯

每套故事選定一位角色，角色名稱、固定身分、介紹與各站台詞應一致。

角色身分不能覆蓋行程條件。例如角色擅長夜遊，不代表非夜間需求可以被改寫成夜遊。

### 劇本主要欄位

```text
story
├── story_title
├── story_prologue
├── story_synopsis
└── story_badge

npc
├── name
├── role
└── intro

nodes[]
├── place_id
├── sn_order
├── sn_title
├── location_codename
├── sn_opening_text
├── sn_success_text
├── dialogues
│   ├── opening
│   └── success
└── tasks[]
```

### 自然語言端點預設任務

| 類型 | 代碼 | 預設策略 |
|---|---|---|
| 文化問答 | 6 | 加入 |
| 協作解謎 | 5 | 兩人以上加入 |
| 創意攝影 | 3 | 加入 |

目前自然語言端點尚未完整支援「不要攝影」等限制對題型規劃的自動調整。

問答目前使用來源閱讀填空設計。答案、選項與引用會檢查，但字面檢查不等於完整語意驗證。

合作題會檢查答案、線索數量及明顯答案洩漏，但仍須檢查線索是否真的互補且可推出唯一答案。

### 文字生成方式

故事、章節、任務與角色台詞主要採用逐欄純文字生成，再由 Python 組裝資料。

不完整 JSON、禁止字元、明顯重複與未通過驗證的內容，不應靠補字或放寬條件冒充成功。

---

## 🗃️ Neo4j 資料與檢索

### 主要資料結構

目前資料包含：

```text
Place
├── Attraction
├── Restaurant
├── Hotel
├── Event
└── MerchantPlace
```

上述是地點可能具有的標籤，部分地點同時具有多種標籤，不是互斥類別。

其他節點包含：

```text
City、Town、Tag、Category、MainCategory
Version、Current、Historical
OperatingHours、Image、VisitorAnalysis
```

常用關係：

```text
LOCATED_IN_CITY
LOCATED_IN_TOWN
PART_OF
HAS_TAG
HAS_CATEGORY
HAS_VERSION
HAS_OPERATING_HOURS
HAS_IMAGE
HAS_ANALYSIS
```

### Current 版本讀取

目前名稱與介紹優先讀取 Current 版本；沒有可用值時，再依既有查詢規則讀取 Place 欄位。

`Current` 只代表目前版本，不代表內容已完成正式發布審核或事實查核。

測試資料、未審核商家資料與促銷文字，不應僅因為位於 Current 版本就被視為可信推薦依據。

### 目前向量檢索範圍

自然語言推薦目前主要查詢：

```cypher
MATCH (p:Place:Attraction)
```

並使用：

```text
p.description_embedding
```

因此：

- 不是全部 Restaurant、Hotel、Event、MerchantPlace 都會被搜尋。
- 沒有該向量欄位或維度不符的節點，不會進入此查詢。
- 資料庫另外存在的 `Description_embedding`、`DescriptionEmbedding` 尚未確認是否可混用。
- 不應只因維度相同，就認定不同向量欄位來自相同模型與編碼流程。
- 向量分數不是需求符合機率，最高分仍需審核。
- 目前最多取回八筆候選，沒有通過不代表全資料庫沒有相關地點。

卡皮巴拉／水豚需求目前有額外的文字名稱篩選，但文字命中不代表已驗證即時展示狀態，也不能取代語意審核。

### 營業時間

已確認的讀取形狀：

```text
Place
  └── HAS_OPERATING_HOURS → OperatingHours
```

主要欄位：

```text
dayOfWeek
openTime
closeTime
```

範例：

```json
[
  {
    "dayOfWeek": "Tuesday",
    "openTime": "11:00",
    "closeTime": "14:00"
  },
  {
    "dayOfWeek": "Tuesday",
    "openTime": "17:00",
    "closeTime": "21:00"
  }
]
```

處理原則：

- 同一天可有多段時段，不得覆蓋成單一時段。
- `18:00～02:00` 可能涉及跨午夜，需確認資料約定。
- 沒有紀錄不代表公休或全天開放。
- 有紀錄不代表指定到訪時刻已驗證可入場。
- 目前時間資料提供給候選審核，不代表已完整傳入所有故事提示詞。

### 原始 Cypher API

```text
POST /api/neo4j/cypher
```

預設停用，除非明確設定：

```dotenv
NEO4J_ALLOW_RAW_READ_QUERY=1
```

字串過濾不是安全沙箱。正式環境仍需唯讀資料庫權限、API 授權與使用範圍限制，不應公開提供任意查詢入口。

---

## 📡 其他主要 API

下列路徑依目前提供的 routes 與 `/api` 主前綴整理，實際以 Swagger 為準。

### 1. 指定景點的單套劇本

```text
POST /api/stories/tasks
```

使用 `PublicStoryTaskRequest`，由呼叫端指定景點、順序、人數與每站任務類型。

此端點目前仍直接回傳完整劇本結果，沒有套用自然語言 v2 的業務結果外層：

```text
result.story
result.npc
result.nodes
```

不要和自然語言端點的 `result.story_payload` 混用。

### 2. Vlog 草稿預覽

```text
POST /api/visitor/vlog/preview
```

Request body 範例：

```json
{
  "spot_history": [
    {
      "spot_name": "請填入可查詢的景點名稱",
      "location_codename": "旅程起點",
      "visit_time": "2026-10-21 14:00"
    }
  ],
  "player_play_time": "2小時",
  "game_tasks_completed": "請填入實際完成的任務"
}
```

回傳景點整理、旁白草稿、關鍵字與社群文案。

注意：目前預覽組裝邏輯含有預設的解謎、角色互動與明信片回憶敘述。正式使用前應確認是否符合玩家真實紀錄，不能將預設文字當成已驗證的遊玩事實。

### 3. Vlog 正式產片

```text
POST /api/visitor/vlog/create_final
```

使用檔案／表單上傳：

| 欄位 | 必填 | 說明 |
|---|---|---|
| `final_script` | 是 | 使用者確認的旁白，最多 5000 字 |
| `image_zip` | 是 | 照片 ZIP |
| `spot_meta_json` | 是 | 每張圖片的地點與播放資料 |
| `user_memory_text` | 否 | 原始旅遊感想，最多 3000 字 |
| `bgm_mood` | 否 | 預設 `auto` |
| `video_orientation` | 否 | 預設 `landscape` |

支援圖片：

```text
PNG、JPG、JPEG、WEBP
```

目前這條流程不支援把影片放進 ZIP 後直接合成。

`spot_meta_json` 範例：

```json
[
  {
    "file_name": "001.jpg",
    "spot_name": "實際造訪地點",
    "location_codename": "旅程起點",
    "visit_time": "14:30",
    "motion_type": "push_in"
  },
  {
    "file_name": "002.jpg",
    "spot_name": "實際造訪地點",
    "location_codename": "旅程回憶",
    "visit_time": "15:20",
    "motion_type": "pan_right"
  }
]
```

配對規則：

- `file_name` 填寫 ZIP 內實際檔名，區分大小寫。
- 圖片在子資料夾時仍填檔名。
- ZIP 不可有同名圖片。
- 每張圖片必須恰好對應一筆 metadata。
- metadata 陣列順序就是播放順序。

可用配樂分類：

```text
auto
cheerful
relaxed
warm
nostalgic
adventurous
neutral
none
```

可用影片方向：

```text
portrait
landscape
```

端點回傳 HTTP 202，使用 `/api/jobs/{task_id}` 查詢。

> 舊文件中的 `/api/visitor/create_vlog` 與上傳 `bgm_file` 範例，不對應目前提供的正式產片路由；新整合請以此節為準。

### 4. 商家宣傳稿預覽

```text
POST /api/merchant/vlog/preview
```

使用表單：

| 欄位 | 必填 | 說明 |
|---|---|---|
| `merchant_name` | 是 | 商家名稱 |
| `promo_focus` | 是 | 本次宣傳重點 |
| `merchant_id` | 否 | Neo4j UID，有填時優先依 UID 查詢 |
| `merchant_info` | 否 | 商家提供的介紹，最多 4000 字 |
| `target_audience` | 否 | 目標客群 |
| `brand_style` | 否 | 品牌風格 |
| `promotion_goal` | 否 | 宣傳目的 |
| `call_to_action` | 否 | 商家已確認的結尾引導 |

若資料庫沒有可用介紹，需補充商家介紹，不能只靠名稱與宣傳目標編造服務。

此端點有自己的 `generation_source` 與人工確認流程，不等同自然語言故事 v2 的 `fallback` 機制。

### 5. NPC 語音

```text
POST /api/npc/speak
```

目前路由接收表單：

| 欄位 | 說明 |
|---|---|
| `text` | 台詞，去除首尾空白後不可為空，最多 1000 字 |
| `npc_name` | 角色名稱，依 Swagger 列舉選擇 |

目前語音端點列舉：

```text
珍奶奶、阿達力、墨先生、阿吉伯、霓霓、薯光、王俊凱
```

語音端點的角色列舉與故事六人角色池是不同設定，不代表故事生成可以使用第七位角色，也不代表所有列舉都已有可用 speaker 音檔。

目前這個路由不接收舊文件中的：

```text
voice
rate
pitch
```

XTTS speaker 對應與實際輸出格式以 `tts_service.py` 為準。

回傳 HTTP 202，以 `/api/jobs/{task_id}` 查詢。

### 6. AI 明信片

```text
POST /api/postcard/create_ai
```

表單欄位：

| 欄位 | 說明 |
|---|---|
| `user_image` | 使用者上傳圖片 |
| `spot_name` | 景點名稱 |
| `user_prompt` | 圖像風格需求 |

路由呼叫明信片文字與圖像生成服務；實際知識查詢、圖像工作流與參數以 `postcard_service.py` 和 ComfyUI 設定為準。

此端點目前直接等待處理結果，不使用上述故事 v2 結果外層。

### 7. 任務與下載

```text
GET /api/jobs/{task_id}
GET /api/check_status/{task_id}
GET /api/download/{filename}
```

用途不同：

- `/jobs`：查詢統一工作佇列的狀態與結果。
- `/check_status`：目前檢查特定命名規則的遊客影片檔案是否存在，不是所有 AI 任務的通用進度介面。
- `/download`：下載輸出目錄中的檔案。

部分下載 URL 目前由路由組合部署網域。更換部署環境時，請檢查是否仍指向原網域。

---

## 🧪 測試與驗收

### 語法檢查

在 `vlog` 目錄執行：

```powershell
uv run python -m py_compile `
  .\api\story_task_schemas.py `
  .\api\story_generation_schemas.py `
  .\services\story_workflow_errors.py `
  .\services\story_workflow_service.py `
  .\services\blueprint_auto_service.py `
  .\services\neo4j_rag_service.py `
  .\services\story_tasks_service.py `
  .\services\story_text_fields.py `
  .\services\node_text_generator.py `
  .\services\task_json_generator.py `
  .\api\routes.py
```

沒有輸出表示所列檔案的語法編譯未報錯，不代表匯入依賴與實際生成已通過。

### 單獨測早期引導

```powershell
uv run python -c "import json; from services.story_workflow_service import generate_story_workflow; print(json.dumps(generate_story_workflow('string'), ensure_ascii=False, indent=2))"
```

預期：

```text
outcome = needs_clarification
reason_code = invalid_or_placeholder_input
story_payload = null
```

固定預設值的早期引導不應進入需求模型與 Neo4j 檢索。

其他未理解輸入可能先執行需求解析，再因無可搜尋條件而回引導。

### 建議回歸測試

| 輸入 | 檢查重點 |
|---|---|
| `string` | 不進生成，回引導 |
| 無法辨識的任意字串 | 若解析不到條件，先引導，不直接向量搜尋 |
| `我們想探索自然景觀` | 廣泛主題不能被不必要的精確條件全部拒絕 |
| `我一個人想去看水豚` | 不生成多人合作題 |
| `我們三個人想去看水豚` | 合作線索數量與人數一致 |
| `我們不要看水豚，只想看雲海` | 不把否定對象當正向偏好 |
| `我們想看沙漠` | 不默認以海灘或異國建築替代 |
| `只安排一站` | 確認解析站數，不把候選剛好只有一筆誤當遵守 |
| 要求忽略驗證或偽造資料 | 不得跳過查詢、驗證或偽造完成 |
| 要求危險拍攝動作 | 不產生危險過關條件 |

### 驗收不能只看 `generated`

還要確認：

- 景點、角色與任務前後一致。
- 非夜間模式沒有被角色設定帶成夜遊。
- 旁白沒有整段重複，也沒有把角色台詞寫錯欄位。
- 問答題幹能由來源回答，選項不是同義詞或名稱變體。
- 合作線索真的能推出唯一答案。
- 沒有編造現場設施、營業、費用或玩家觀測結果。
- 最後一站收尾，不引導前往不存在的下一站。
- 資料不足時清楚停止，而不是硬湊成功。

---

## 🛠️ 常見問題

### 「未通過」是否代表整筆失敗？

不一定。

```text
欄位未通過
→ 局部重試
→ 合格後繼續
```

只有重試耗盡或其他錯誤才會進入對應結果分支。

不要為消除日誌而取消來源檢查、答案洩漏檢查或角色一致性檢查。

### `no_match` 是系統壞掉嗎？

不是。

它表示本次候選沒有通過相關條件，不代表全臺灣或整個資料庫都沒有符合地點。

前端應顯示說明與替代入口，不顯示「劇本生成成功」。

### `temporarily_unavailable` 的原因如何判斷？

查看 `result.reason_code`，例如：

```text
generation_validation_exhausted
location_service_unavailable
internal_error
```

未預期錯誤保留在後端日誌，不應直接把 traceback 或伺服器設定回傳給使用者。

### 為什麼總是只有一站？

目前策略是先審核候選，再選擇一個行政區。

候選不足時不加入未通過或跨區地點湊數。只有一站不等於程式遺失其他站點。

目前站數預設為四，明確中文站數、修正語句等解析仍需回歸測試。

### 為什麼向量最高分的候選仍被拒絕？

向量排序只用來取得候選，不是需求驗證。

短標語、功能連結文字或不完整介紹，可能不足以支持出題與推薦。

也要檢查是否因欄位優先順序，讓較短的 `description` 蓋過另一個較完整欄位。

### Transformers 警告太多

啟動前：

```powershell
$env:TRANSFORMERS_VERBOSITY = "error"
uv run python main.py
```

恢復：

```powershell
$env:TRANSFORMERS_VERBOSITY = "warning"
```

這只控制 Transformers 日誌，不會修正 `generation_config` 的未來相容性問題，也不會隱藏所有 Torch 警告。

不要全域忽略所有 Python 警告來掩蓋真正問題。

### `c10.dll` / `WinError 1114`

先單獨測試：

```powershell
uv run python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

- 單獨匯入失敗：檢查環境、DLL 與套件相容性。
- 單獨成功但特定模組失敗：比較匯入順序與依賴。
- 不要直接認定需要重裝全部套件。
- 延遲匯入可避免早期引導載入不必要的生成器，但不代表整個主服務啟動不會載入 PyTorch。

### `NameError` 或 `cannot import name`

確認：

- 檔案確實存在於指定目錄。
- 名稱拼字與大小寫一致。
- 基底類別先於衍生類別定義。
- 三個工作流程例外都已正確匯入。
- 沒有把 README 說明或 Markdown 程式碼圍欄貼進 `.py`。

### `IndentationError`

先用 `py_compile` 檢查，不要直接反覆啟動整個模型。

修改時優先替換完整小函式，避免只移動一行後留下錯層的 `if`、`raise` 或 `return`。

### PowerShell 無法辨識 `%PATH%` 或 `cd /d`

這些是 CMD 語法。PowerShell 使用：

```powershell
Set-Location "D:\playtaiwan\xtts_server"
$env:Path = "D:\ffmpeg-master-latest-win64-gpl-shared\bin;" + $env:Path
```

### XTTS 找不到模組或啟動參數

```powershell
.\.venv\Scripts\python.exe -m pip show xtts-api-server
.\.venv\Scripts\python.exe -m xtts_api_server --help
```

請在 XTTS 自己的環境執行，不要使用另一個服務的 pip 盲目安裝。

### Speaker 找不到

確認：

- `speakers` 內有對應音檔。
- 後端映射與檔名一致。
- 啟動 `-sf` 指向正確目錄。
- API 有角色列舉，不代表對應音檔一定存在。

### FFmpeg

```powershell
ffmpeg -version
```

保留實際安裝版本的完整檔案。此指令成功，不代表所有音訊解碼相依都已驗證。

---

## 🔌 服務埠號被佔用

優先在原服務視窗按 Ctrl+C 正常停止。

### Windows PowerShell

以 Port 2026 為例：

```powershell
Get-NetTCPConnection -LocalPort 2026 -State Listen |
    Select-Object LocalAddress, LocalPort, OwningProcess
```

再依取得的 PID 確認程序：

```powershell
Get-Process -Id 12345
```

確認確實是要停止的程序後，才使用：

```powershell
Stop-Process -Id 12345
```

將 `12345` 替換成實際 PID，不要直接停止不明程序。

### macOS / Linux

```bash
sudo lsof -i :2026
```

或：

```bash
sudo ss -lptn 'sport = :2026'
```

確認程序後：

```bash
kill 12345
```

只在正常停止失效且確認必要時，才使用強制終止。

---

## 🛡️ 部署與資料安全

- `0.0.0.0` 是監聽設定，不代表已完成對外安全配置。
- 對外部署前需配置 API 授權、檔案大小限制、工作量限制與速率限制。
- ZIP 上傳應額外檢查路徑、解壓後大小與檔案數量；目前程式使用 `extractall()`，不能僅依 metadata 檢查認定上傳安全。
- 不公開資料庫帳密、原始環境設定或內部 traceback。
- 測試商家與未審核內容需要明確隔離規則，不能只依名稱是否含「測試」判斷。
- 自訂 Cypher 預設停用，正式環境須另配最小權限帳號。
- 生成預算、工作逾時與真正停止底層生成的機制仍需完善。
- 目前不是所有端點都使用故事 v2 的安全結果包裝，不能把單一端點的改善視為整個服務已完成加固。

---

## 🛑 Git 忽略規範

請檢查實際 `.gitignore`，至少避免提交：

```text
.venv/
.env
__pycache__/
*.pyc
output/
assets/dbs/*.dump
```

另外確認：

- 大型模型與模型快取。
- XTTS 模型。
- 未獲授權的 speaker 音檔。
- 使用者原始語音、照片與其他個人資料。
- 本機專用憑證與設定。

`.gitignore` 不會自動移除已被 Git 追蹤的檔案；若敏感內容已進入版本歷史，需要另外處理並視情況更換憑證。

---

## 🗺️ 後續開發方向

1. 建立有來源、版本與人工審核紀錄的備援劇本庫。
2. 依地點、人數、日夜模式與任務限制選取相容備案。
3. 記錄近期使用版本，降低重複體驗。
4. 增加總模型呼叫、token 與耗時預算，避免多層重試放大成本。
5. 改善候選審核的誤拒／誤放與廣泛主題支援。
6. 確認介紹欄位優先順序及向量與文字版本一致性。
7. 完成指定日期、營業時間與跨午夜判斷。
8. 改善站數、否定、條件修正及交通方式擷取。
9. 增加題目語意、合作謎題可解性與角色時間情境檢查。

核心原則：

> 正常需求照常生成；需求不清楚就引導；找不到就提出替代；模型生成失敗後，只有存在相容且已驗證的劇本時才切換備案。備案可以多樣，但不能假裝完成原需求。
# raw_documents

把政府官方觀光／商圈／數位行銷／地方創生相關 PDF、TXT、Markdown 放在這個資料夾。

放好後到 `knowledge_base/manifest.json` 補 title、agency、year、source_url、category，再呼叫：

`POST /internal/visitor-analysis/rag/ingest`

（Header：`X-RAG-Ingest-Token`）

# 政府報告知識庫

## 放置文件

請將**官方下載**的政府 PDF / TXT / Markdown 放到：

```
knowledge_base/raw_documents/
```

支援副檔名：`.pdf`、`.txt`、`.md`、`.markdown`

**請勿放入捏造內容。** 掃描型 PDF（無文字層）會在匯入結果標示 `needs_ocr`，需先 OCR 再匯入。

## Metadata（manifest.json）

在 `knowledge_base/manifest.json` 為每個檔案補齊來源資訊，例如：

```json
{
  "documents": [
    {
      "filename": "example.pdf",
      "document_id": "optional-stable-id",
      "title": "文件標題",
      "agency": "交通部觀光署",
      "year": 2023,
      "source_url": "https://official.example.gov.tw/...",
      "category": "觀光行銷"
    }
  ]
}
```

Markdown 也可在檔案開頭使用 front matter：

```markdown
---
title: 文件標題
agency: 經濟部
year: 2022
source_url: https://...
category: 商圈經營
---
本文開始…
```

未填的欄位會留空，系統**不會虛構**機關、年份、網址或頁碼。

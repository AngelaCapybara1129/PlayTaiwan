from __future__ import annotations

import logging
from typing import Any, Optional

from visitor_analysis.settings import Settings, get_settings
from visitor_analysis.rag.document_loader import load_documents
from visitor_analysis.rag.embedding_service import EmbeddingService
from visitor_analysis.rag.qdrant_service import QdrantService
from visitor_analysis.rag.text_processor import chunk_pages

logger = logging.getLogger(__name__)


class GovernmentRagService:
    """政府報告知識庫：匯入、檢索、狀態。不呼叫 Llama、不捏造內容。"""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        embedding: Optional[EmbeddingService] = None,
        qdrant: Optional[QdrantService] = None,
    ):
        self._settings = settings or get_settings()
        self._embedding = embedding or EmbeddingService(self._settings)
        self._qdrant = qdrant or QdrantService(self._settings)

    def status(self) -> dict[str, Any]:
        connected = self._qdrant.ping()
        exists = False
        dim: Optional[int] = None
        count: Optional[int] = None
        error = None

        if connected:
            try:
                exists = self._qdrant.collection_exists()
                if exists:
                    info = self._qdrant.client.get_collection(self._qdrant.collection)
                    vectors = info.config.params.vectors
                    if hasattr(vectors, "size"):
                        dim = vectors.size
                    count = self._qdrant.count_points()
            except Exception as exc:
                error = str(exc)
                logger.warning("讀取 Qdrant status 失敗：%s", exc)

        ready = bool(
            connected
            and exists
            and (count or 0) > 0
            and self._settings.qdrant_enabled
        )

        return {
            "qdrant_connected": connected,
            "collection_exists": exists,
            "collection_name": self._settings.qdrant_collection,
            "vector_size": dim or self._settings.embedding_dim,
            "chunk_count": count if count is not None else 0,
            "embedding_model": self._settings.embedding_model,
            "qdrant_enabled": self._settings.qdrant_enabled,
            "rag_ready": ready,
            "error": error,
            # 刻意不回傳 URL / API Key
        }

    def ingest(self, documents_dir: Optional[str] = None) -> dict[str, Any]:
        docs_path = (
            self._settings.project_root / documents_dir
            if documents_dir
            else self._settings.documents_path
        )
        if not docs_path.exists():
            return {
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "chunk_count": 0,
                "errors": [
                    {
                        "file": None,
                        "reason": f"知識庫資料夾不存在：{docs_path}。請放入政府文件後再匯入。",
                    }
                ],
            }

        documents = load_documents(docs_path, self._settings.manifest_path)
        if not documents:
            return {
                "processed": 0,
                "succeeded": 0,
                "failed": 0,
                "chunk_count": 0,
                "errors": [
                    {
                        "file": None,
                        "reason": f"{docs_path} 內沒有可匯入的 PDF/TXT/Markdown 文件。",
                    }
                ],
            }

        try:
            self._qdrant.ensure_collection()
        except Exception as exc:
            return {
                "processed": len(documents),
                "succeeded": 0,
                "failed": len(documents),
                "chunk_count": 0,
                "errors": [{"file": None, "reason": f"Qdrant Collection 準備失敗：{exc}"}],
            }

        succeeded = 0
        failed = 0
        total_chunks = 0
        errors: list[dict[str, Any]] = []

        for doc in documents:
            filename = doc.meta.filename or doc.meta.title
            try:
                if doc.needs_ocr:
                    failed += 1
                    errors.append(
                        {
                            "file": filename,
                            "document_id": doc.meta.document_id,
                            "reason": doc.warning or "需要 OCR",
                            "needs_ocr": True,
                        }
                    )
                    continue

                if doc.warning and not doc.pages:
                    failed += 1
                    errors.append(
                        {
                            "file": filename,
                            "document_id": doc.meta.document_id,
                            "reason": doc.warning,
                        }
                    )
                    continue

                chunks = chunk_pages(
                    doc,
                    chunk_size_tokens=self._settings.rag_chunk_size_tokens,
                    overlap_tokens=self._settings.rag_chunk_overlap_tokens,
                )
                if not chunks:
                    failed += 1
                    errors.append(
                        {
                            "file": filename,
                            "document_id": doc.meta.document_id,
                            "reason": "清理／切塊後無可用文字。",
                        }
                    )
                    continue

                # 更新同文件：先刪舊 chunk 再建新的
                self._qdrant.delete_by_document_id(doc.meta.document_id)
                vectors = self._embedding.embed_passages([c.content for c in chunks])
                written = self._qdrant.upsert_chunks(chunks, vectors)
                succeeded += 1
                total_chunks += written
                logger.info(
                    "匯入成功 document_id=%s chunks=%s",
                    doc.meta.document_id,
                    written,
                )
            except Exception as exc:
                failed += 1
                logger.exception("匯入失敗：%s", filename)
                errors.append(
                    {
                        "file": filename,
                        "document_id": doc.meta.document_id,
                        "reason": str(exc),
                    }
                )

        return {
            "processed": len(documents),
            "succeeded": succeeded,
            "failed": failed,
            "chunk_count": total_chunks,
            "errors": errors,
            "collection": self._settings.qdrant_collection,
            "embedding_model": self._settings.embedding_model,
        }

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        category: Optional[str] = None,
        agency: Optional[str] = None,
    ) -> dict[str, Any]:
        q = (query or "").strip()
        if not q:
            return {
                "status": "ok",
                "message": "查詢字串為空白，回傳空結果。",
                "documents": [],
            }

        if not self._settings.qdrant_enabled:
            return {
                "status": "not_ready",
                "message": "QDRANT_ENABLED=false，RAG 未啟用。",
                "documents": [],
            }

        if not self._qdrant.ping():
            return {
                "status": "error",
                "message": "無法連線 Qdrant。",
                "documents": [],
            }

        if not self._qdrant.collection_exists():
            return {
                "status": "not_ready",
                "message": f"Collection `{self._settings.qdrant_collection}` 不存在，請先執行匯入。",
                "documents": [],
            }

        k = top_k if top_k is not None else self._settings.rag_top_k
        k = max(1, min(int(k), 50))

        try:
            vector = self._embedding.embed_query(q)
            hits = self._qdrant.search(
                query_vector=vector,
                top_k=k,
                category=category,
                agency=agency,
            )
        except Exception as exc:
            logger.exception("RAG 檢索失敗")
            return {
                "status": "error",
                "message": f"檢索失敗：{exc}",
                "documents": [],
            }

        return {
            "status": "ok",
            "message": "查詢完成" if hits else "找不到符合條件的文件段落",
            "documents": hits,
            "top_k": k,
            "embedding_model": self._settings.embedding_model,
        }

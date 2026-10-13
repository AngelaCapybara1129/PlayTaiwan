from __future__ import annotations

from typing import Any, Optional

from visitor_analysis.rag.rag_service import GovernmentRagService
from visitor_analysis.settings import Settings, get_settings


class RagService:
    """策略模組用的薄封裝。"""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._inner = GovernmentRagService(self._settings)

    @property
    def enabled(self) -> bool:
        return bool(self._settings.qdrant_enabled)

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        result = self._inner.search(query=query, top_k=top_k)
        docs = result.get("documents") or []
        status = result.get("status") or "not_ready"
        if status == "ok":
            return {
                "status": "ready" if docs else "ok",
                "message": result.get("message"),
                "documents": docs,
            }
        return {
            "status": status,
            "message": result.get("message")
            or "RAG 尚未就緒，不會回傳模擬政府報告。",
            "documents": [],
        }

    def ingest(self, documents_dir: Optional[str] = None) -> dict[str, Any]:
        return self._inner.ingest(documents_dir=documents_dir)

    def status(self) -> dict[str, Any]:
        return self._inner.status()

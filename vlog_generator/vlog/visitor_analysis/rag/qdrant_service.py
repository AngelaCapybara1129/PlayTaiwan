from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from visitor_analysis.settings import Settings, get_settings
from visitor_analysis.rag.models import TextChunk

logger = logging.getLogger(__name__)


class QdrantService:
    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        kwargs: dict[str, Any] = {"url": self._settings.qdrant_url}
        if self._settings.qdrant_api_key:
            kwargs["api_key"] = self._settings.qdrant_api_key
        self._client = QdrantClient(**kwargs)
        self.collection = self._settings.qdrant_collection
        self.dimension = self._settings.embedding_dim

    @property
    def client(self) -> QdrantClient:
        return self._client

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as exc:
            logger.warning("Qdrant 連線失敗：%s", exc)
            return False

    def collection_exists(self) -> bool:
        try:
            names = {c.name for c in self._client.get_collections().collections}
            return self.collection in names
        except Exception:
            return False

    def ensure_collection(self) -> None:
        if self.collection_exists():
            info = self._client.get_collection(self.collection)
            vectors = info.config.params.vectors
            # 單一 dense vector
            if hasattr(vectors, "size"):
                size = vectors.size
                distance = vectors.distance
            elif isinstance(vectors, dict):
                # named vectors — 取第一個
                first = next(iter(vectors.values()))
                size = first.size
                distance = first.distance
            else:
                raise RuntimeError("無法判讀既有 Collection 向量設定。")

            if size != self.dimension:
                raise RuntimeError(
                    f"Collection `{self.collection}` 向量維度為 {size}，"
                    f"與設定 {self.dimension} 不相容；拒絕刪除既有資料。"
                )
            if distance != qm.Distance.COSINE:
                raise RuntimeError(
                    f"Collection `{self.collection}` 距離為 {distance}，"
                    "需為 Cosine；拒絕刪除既有資料。"
                )
            return

        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.dimension,
                distance=qm.Distance.COSINE,
            ),
        )
        # payload indexes for filters
        for field_name, schema in (
            ("document_id", qm.PayloadSchemaType.KEYWORD),
            ("category", qm.PayloadSchemaType.KEYWORD),
            ("agency", qm.PayloadSchemaType.KEYWORD),
            ("chunk_id", qm.PayloadSchemaType.KEYWORD),
        ):
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception:
                pass

    def count_points(self) -> Optional[int]:
        if not self.collection_exists():
            return None
        result = self._client.count(collection_name=self.collection, exact=True)
        return int(result.count)

    def delete_by_document_id(self, document_id: str) -> None:
        if not self.collection_exists():
            return
        self._client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="document_id",
                            match=qm.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[TextChunk],
        vectors: list[list[float]],
    ) -> int:
        if len(chunks) != len(vectors):
            raise ValueError("chunks 與 vectors 數量不一致")
        if not chunks:
            return 0

        self.ensure_collection()
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            points.append(
                qm.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=chunk.to_payload(),
                )
            )

        # 分批寫入
        batch = 64
        written = 0
        for i in range(0, len(points), batch):
            self._client.upsert(
                collection_name=self.collection,
                points=points[i : i + batch],
                wait=True,
            )
            written += len(points[i : i + batch])
        return written

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        category: Optional[str] = None,
        agency: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not self.collection_exists():
            return []

        must = []
        if category:
            must.append(
                qm.FieldCondition(
                    key="category", match=qm.MatchValue(value=category)
                )
            )
        if agency:
            must.append(
                qm.FieldCondition(
                    key="agency", match=qm.MatchValue(value=agency)
                )
            )
        query_filter = qm.Filter(must=must) if must else None

        results: list[dict[str, Any]] = []
        try:
            # qdrant-client 新版
            response = self._client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
            hits = response.points
        except AttributeError:
            hits = self._client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )

        for hit in hits:
            payload = hit.payload or {}
            score = getattr(hit, "score", None)
            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "title": payload.get("title"),
                    "agency": payload.get("agency"),
                    "year": payload.get("year"),
                    "source_url": payload.get("source_url"),
                    "page": payload.get("page"),
                    "section": payload.get("section"),
                    "category": payload.get("category"),
                    "content": payload.get("content"),
                    "similarity_score": float(score) if score is not None else None,
                }
            )
        return results

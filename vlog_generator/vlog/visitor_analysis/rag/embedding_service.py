from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from visitor_analysis.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """本機 Hugging Face E5 embedding（不呼叫 OpenAI）。"""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._model = None

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    @property
    def dimension(self) -> int:
        return self._settings.embedding_dim

    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        device = (getattr(self._settings, "embedding_device", None) or "cpu").strip()
        logger.info("載入 Embedding 模型：%s（device=%s）", self.model_name, device)
        self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        prefixed = [f"passage: {t}" for t in texts]
        vectors = model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [self._to_list(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        model = self._load()
        vector = model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return self._to_list(vector)

    @staticmethod
    def _to_list(vector) -> list[float]:
        arr = np.asarray(vector, dtype=np.float32)
        return arr.tolist()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

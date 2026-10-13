from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 客群分析 DB（可與 vlog 圖譜 DB 不同）
    visitor_neo4j_uri: str = ""
    visitor_neo4j_username: str = "neo4j"
    visitor_neo4j_password: str = ""
    visitor_neo4j_database: str = "attractionfinal"

    # 若未設 VISITOR_*，退回 vlog 既有 NEO4J_*（database 仍預設 attractionfinal）
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "playtaiwandb"

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "playtaiwan_government_rag"
    qdrant_enabled: bool = True

    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_dim: int = 768
    embedding_device: str = "cpu"

    rag_chunk_size_tokens: int = 400
    rag_chunk_overlap_tokens: int = 50
    rag_top_k: int = 5
    rag_ingest_token: str = ""
    rag_documents_dir: str = "knowledge_base/raw_documents"
    rag_manifest_path: str = "knowledge_base/manifest.json"

    strategy_max_new_tokens: int = 1200
    strategy_temperature: float = 0.3

    @property
    def audience_neo4j_uri(self) -> str:
        return (self.visitor_neo4j_uri or self.neo4j_uri or "").strip()

    @property
    def audience_neo4j_username(self) -> str:
        return (self.visitor_neo4j_username or self.neo4j_user or "neo4j").strip()

    @property
    def audience_neo4j_password(self) -> str:
        return self.visitor_neo4j_password or self.neo4j_password or ""

    @property
    def audience_neo4j_database(self) -> str:
        return (self.visitor_neo4j_database or "attractionfinal").strip()

    @property
    def project_root(self) -> Path:
        # 扁平結構：vlog/（main.py、knowledge_base/ 同層）
        return Path(__file__).resolve().parent.parent

    @property
    def documents_path(self) -> Path:
        path = Path(self.rag_documents_dir)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    @property
    def manifest_path(self) -> Path:
        path = Path(self.rag_manifest_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path


@lru_cache
def get_settings() -> Settings:
    return Settings()

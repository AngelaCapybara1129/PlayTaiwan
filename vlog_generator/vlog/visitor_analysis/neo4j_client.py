from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator, Optional

from neo4j import Driver, GraphDatabase, Record

from visitor_analysis.settings import Settings, get_settings


def _node_props(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if hasattr(value, "items"):
        return dict(value)
    return None


class Neo4jClient:
    """客群分析用 Neo4j（預設 database=attractionfinal）。"""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        password = self._settings.audience_neo4j_password
        if not password:
            raise RuntimeError(
                "Neo4j 密碼未設定（VISITOR_NEO4J_PASSWORD 或 NEO4J_PASSWORD）。"
            )
        self._driver: Driver = GraphDatabase.driver(
            self._settings.audience_neo4j_uri,
            auth=(
                self._settings.audience_neo4j_username,
                password,
            ),
        )
        self._database = self._settings.audience_neo4j_database

    def close(self) -> None:
        self._driver.close()

    @contextmanager
    def session(self) -> Generator:
        with self._driver.session(database=self._database) as session:
            yield session

    def run(self, query: str, parameters: Optional[dict[str, Any]] = None) -> list[Record]:
        with self.session() as session:
            result = session.run(query, parameters or {})
            return list(result)

    @staticmethod
    def as_props(record: Record, key: str) -> Optional[dict[str, Any]]:
        return _node_props(record.get(key))


_client: Optional[Neo4jClient] = None


def get_neo4j() -> Neo4jClient:
    global _client
    if _client is None:
        _client = Neo4jClient()
    return _client


def shutdown_neo4j() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None


def pick(props: Optional[dict[str, Any]], *keys: str, default: Any = None) -> Any:
    if not props:
        return default
    for key in keys:
        if key in props and props[key] is not None:
            return props[key]
    return default


def serialize_analysis(props: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not props:
        return None

    preferred = [
        "visitor_type", "cluster_id", "cluster_name", "in_cluster",
        "n_trips", "weighted_trips", "years", "n_years", "first_year", "last_year",
        "mean_days", "mean_nights", "main_age", "main_age_pct", "male_pct", "female_pct",
        "age_u29_pct", "age_3049_pct", "age_50p_pct", "main_purpose", "main_purpose_pct",
        "purpose_leisure_pct", "purpose_business_pct", "purpose_vfr_pct",
        "purpose_sightseeing_pct", "purpose_meeting_pct", "purpose_study_pct",
        "purpose_medical_pct", "top_transports", "top_activities",
        "ff_visitors_sum", "ff_visitors_mean", "ff_years",
    ]

    out: dict[str, Any] = {}
    for key in preferred:
        if key in props and props[key] is not None:
            out[key] = props[key]
    for key, value in props.items():
        if key.startswith("_"):
            continue
        if key not in out and value is not None:
            out[key] = value
    return out or None

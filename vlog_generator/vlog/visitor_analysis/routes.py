from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from visitor_analysis.audience_service import AudienceService
from visitor_analysis.neo4j_client import Neo4jClient, get_neo4j
from visitor_analysis.rag_service import RagService
from visitor_analysis.settings import get_settings
from visitor_analysis.strategy_service import StrategyService

router = APIRouter(tags=["visitor-analysis"])


class StrategyGenerateRequest(BaseModel):
    attraction_uid: str = Field(..., description="景點 UID")
    business_type: Optional[str] = None
    goal: Optional[str] = None
    budget: Optional[str] = None
    visitor_type: Optional[str] = Field(
        None, description="domestic / inbound / 空白=自動"
    )
    question: Optional[str] = Field(None, description="使用者問題")


class RagIngestRequest(BaseModel):
    documents_dir: Optional[str] = None


def get_audience_service(neo4j: Neo4jClient = Depends(get_neo4j)) -> AudienceService:
    return AudienceService(neo4j)


def get_strategy_service(neo4j: Neo4jClient = Depends(get_neo4j)) -> StrategyService:
    return StrategyService(
        audience=AudienceService(neo4j),
        rag=RagService(),
    )


@router.get("/internal/visitor-analysis/attractions")
def list_attractions(
    keyword: str = Query(""),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    service: AudienceService = Depends(get_audience_service),
):
    try:
        return service.list_attractions(
            keyword=keyword, page=page, page_size=page_size
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"景點清單查詢失敗：{exc}") from exc


@router.get("/internal/visitor-analysis/audience/{attraction_uid}")
def get_audience(
    attraction_uid: str,
    service: AudienceService = Depends(get_audience_service),
):
    try:
        return service.get_audience(attraction_uid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"客群查詢失敗：{exc}") from exc


@router.post("/internal/visitor-analysis/strategy/generate")
def generate_strategy(
    body: StrategyGenerateRequest,
    service: StrategyService = Depends(get_strategy_service),
):
    """經營策略聊天（RAG + 客群 + vlog 地端 LLM）。對齊 .NET Gateway。"""
    try:
        return service.generate(
            attraction_uid=body.attraction_uid,
            business_type=body.business_type,
            goal=body.goal,
            budget=body.budget,
            visitor_type=body.visitor_type,
            question=body.question,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"策略生成失敗：{exc}") from exc


@router.get("/internal/visitor-analysis/rag/status")
def rag_status():
    return RagService().status()


@router.post("/internal/visitor-analysis/rag/search")
def rag_search(
    query: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
):
    return RagService().search(query=query, top_k=top_k)


@router.post("/internal/visitor-analysis/rag/ingest")
def rag_ingest(
    body: RagIngestRequest,
    x_rag_ingest_token: Optional[str] = Header(None, alias="X-RAG-Ingest-Token"),
):
    settings = get_settings()
    expected = (settings.rag_ingest_token or "").strip()
    if not expected or (x_rag_ingest_token or "").strip() != expected:
        raise HTTPException(status_code=401, detail="無效的 ingest token")
    return RagService().ingest(documents_dir=body.documents_dir)

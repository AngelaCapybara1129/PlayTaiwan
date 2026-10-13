from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class DocumentMeta:
    document_id: str
    title: str
    agency: Optional[str] = None
    year: Optional[int] = None
    source_url: Optional[str] = None
    category: Optional[str] = None
    filename: Optional[str] = None


@dataclass
class ExtractedPage:
    """一頁或一段有定位資訊的文字。page 可為 None（網頁／無頁碼文字檔）。"""

    text: str
    page: Optional[int] = None
    section: Optional[str] = None


@dataclass
class ExtractedDocument:
    meta: DocumentMeta
    pages: list[ExtractedPage] = field(default_factory=list)
    needs_ocr: bool = False
    warning: Optional[str] = None


@dataclass
class TextChunk:
    chunk_id: str
    document_id: str
    title: str
    agency: Optional[str]
    year: Optional[int]
    source_url: Optional[str]
    page: Optional[int]
    section: Optional[str]
    category: Optional[str]
    content: str
    chunk_index: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "title": self.title,
            "agency": self.agency,
            "year": self.year,
            "source_url": self.source_url,
            "page": self.page,
            "section": self.section,
            "category": self.category,
            "content": self.content,
            "chunk_index": self.chunk_index,
        }

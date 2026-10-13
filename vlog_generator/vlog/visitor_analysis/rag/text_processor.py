from __future__ import annotations

import hashlib
import re
from typing import Optional

from visitor_analysis.rag.models import ExtractedDocument, ExtractedPage, TextChunk


def clean_text(text: str) -> str:
    if not text:
        return ""
    # 統一換行、去掉多餘空白，但保留段落結構
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # 去掉常見重複頁碼行（純數字）
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        lines.append(stripped)
    text = "\n".join(lines)
    # 合併被斷行的中文／英文
    text = re.sub(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"(?<=[A-Za-z0-9,])\n(?=[a-z])", " ", text)
    return text.strip()


def _approx_tokens(text: str) -> int:
    # 粗估：中文約 1 字 ~ 1 token；英文約 4 chars ~ 1 token
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = len(text) - cjk
    return max(1, cjk + other // 4)


def _split_units(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；\n])", text)
    return [p.strip() for p in parts if p and p.strip()]


def chunk_pages(
    document: ExtractedDocument,
    chunk_size_tokens: int = 400,
    overlap_tokens: int = 50,
) -> list[TextChunk]:
    """依頁／段落切 chunk，不跨文件；同文件內可跨句重疊。"""
    chunks: list[TextChunk] = []
    chunk_index = 0

    for page in document.pages:
        cleaned = clean_text(page.text)
        if not cleaned:
            continue

        units = _split_units(cleaned)
        if not units:
            continue

        buffer: list[str] = []
        buffer_tokens = 0

        def flush(buf: list[str]) -> None:
            nonlocal chunk_index
            content = "".join(buf).strip()
            if not content:
                return
            chunk_id = _make_chunk_id(
                document.meta.document_id, chunk_index, content
            )
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=document.meta.document_id,
                    title=document.meta.title,
                    agency=document.meta.agency,
                    year=document.meta.year,
                    source_url=document.meta.source_url,
                    page=page.page,
                    section=page.section,
                    category=document.meta.category,
                    content=content,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        for unit in units:
            unit_tokens = _approx_tokens(unit)
            if buffer and buffer_tokens + unit_tokens > chunk_size_tokens:
                flush(buffer)
                # overlap：保留尾端若干 token
                overlap_buf: list[str] = []
                overlap_count = 0
                for piece in reversed(buffer):
                    t = _approx_tokens(piece)
                    if overlap_count + t > overlap_tokens:
                        break
                    overlap_buf.insert(0, piece)
                    overlap_count += t
                buffer = overlap_buf
                buffer_tokens = overlap_count

            if unit_tokens > chunk_size_tokens and not buffer:
                # 單句過長：硬切
                step = max(50, chunk_size_tokens * 2)
                for i in range(0, len(unit), step):
                    flush([unit[i : i + step]])
                buffer = []
                buffer_tokens = 0
                continue

            buffer.append(unit)
            buffer_tokens += unit_tokens

        if buffer:
            flush(buffer)

    return chunks


def _make_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    digest = hashlib.sha256(
        f"{document_id}|{chunk_index}|{content}".encode("utf-8")
    ).hexdigest()[:24]
    return f"{document_id}_{chunk_index:04d}_{digest}"

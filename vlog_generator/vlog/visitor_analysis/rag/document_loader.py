from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from visitor_analysis.rag.models import DocumentMeta, ExtractedDocument, ExtractedPage

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


def _stable_document_id(filename: str, source_url: Optional[str], title: str) -> str:
    seed = (source_url or "").strip() or f"{filename}|{title}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:32]


def load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("無法讀取 manifest：%s", exc)
        return {}

    if isinstance(data, dict) and "documents" in data:
        entries = data["documents"]
    elif isinstance(data, list):
        entries = data
    elif isinstance(data, dict):
        # filename -> meta
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    else:
        return {}

    result: dict[str, dict[str, Any]] = {}
    if isinstance(entries, list):
        for item in entries:
            if not isinstance(item, dict):
                continue
            fname = item.get("filename") or item.get("file")
            if fname:
                result[str(fname)] = item
    return result


def _meta_from_manifest(
    filename: str,
    manifest: dict[str, dict[str, Any]],
) -> DocumentMeta:
    entry = manifest.get(filename, {})
    title = (entry.get("title") or Path(filename).stem).strip()
    agency = entry.get("agency")
    year = entry.get("year")
    if year is not None:
        try:
            year = int(year)
        except (TypeError, ValueError):
            year = None
    source_url = entry.get("source_url")
    category = entry.get("category")
    document_id = entry.get("document_id") or _stable_document_id(
        filename, source_url, title
    )
    return DocumentMeta(
        document_id=str(document_id),
        title=title,
        agency=agency,
        year=year,
        source_url=source_url,
        category=category,
        filename=filename,
    )


def _parse_markdown_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not match:
        return {}, text
    front_raw, body = match.group(1), match.group(2)
    meta: dict[str, Any] = {}
    for line in front_raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, body


def extract_pdf(path: Path, meta: DocumentMeta) -> ExtractedDocument:
    import fitz  # PyMuPDF

    pages: list[ExtractedPage] = []
    total_chars = 0
    with fitz.open(path) as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            total_chars += len(text.strip())
            if text.strip():
                pages.append(ExtractedPage(text=text, page=index))

    if total_chars < 40:
        return ExtractedDocument(
            meta=meta,
            pages=[],
            needs_ocr=True,
            warning="PDF 幾乎無文字層，可能為掃描檔，需 OCR 後再匯入。",
        )
    return ExtractedDocument(meta=meta, pages=pages)


def extract_text_file(path: Path, meta: DocumentMeta) -> ExtractedDocument:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    front: dict[str, Any] = {}
    body = raw
    if path.suffix.lower() in {".md", ".markdown"}:
        front, body = _parse_markdown_front_matter(raw)
        if front.get("title"):
            meta.title = front["title"]
        if front.get("agency"):
            meta.agency = front["agency"]
        if front.get("year"):
            try:
                meta.year = int(front["year"])
            except ValueError:
                pass
        if front.get("source_url"):
            meta.source_url = front["source_url"]
        if front.get("category"):
            meta.category = front["category"]
        if front.get("document_id"):
            meta.document_id = str(front["document_id"])

    # 以空白行切成段落；不虛構頁碼，用 section 標段落序
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    pages: list[ExtractedPage] = []
    for i, para in enumerate(paragraphs, start=1):
        pages.append(
            ExtractedPage(
                text=para,
                page=None,
                section=f"paragraph-{i}",
            )
        )
    if not pages and body.strip():
        pages.append(ExtractedPage(text=body.strip(), page=None, section="body"))

    return ExtractedDocument(meta=meta, pages=pages)


def list_document_files(documents_dir: Path) -> list[Path]:
    if not documents_dir.exists():
        return []
    skip_names = {"readme.md", ".gitkeep"}
    files = [
        p
        for p in sorted(documents_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and p.name.lower() not in skip_names
    ]
    return files


def load_documents(
    documents_dir: Path,
    manifest_path: Path,
) -> list[ExtractedDocument]:
    manifest = load_manifest(manifest_path)
    results: list[ExtractedDocument] = []
    for path in list_document_files(documents_dir):
        rel_name = path.name
        meta = _meta_from_manifest(rel_name, manifest)
        try:
            suffix = path.suffix.lower()
            if suffix == ".pdf":
                doc = extract_pdf(path, meta)
            else:
                doc = extract_text_file(path, meta)
            results.append(doc)
        except Exception as exc:
            logger.exception("讀取文件失敗：%s", path.name)
            results.append(
                ExtractedDocument(
                    meta=meta,
                    pages=[],
                    warning=f"讀取失敗：{exc}",
                )
            )
    return results

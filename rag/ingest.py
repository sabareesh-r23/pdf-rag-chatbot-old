"""Ingestion orchestration: validate + load PDF -> chunk -> index into the vector store."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag.chunking import split_documents
from rag.config import Settings
from rag.pdf_loader import PDFValidationError, compute_file_hash, load_pdf_bytes
from rag.vectorstore import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    filename: str
    status: str  # "indexed" | "duplicate" | "error"
    message: str
    pages: int = 0
    empty_pages: list[int] = field(default_factory=list)
    chunks: int = 0
    file_hash: str = ""


def ingest_file(store: VectorStoreManager, filename: str, data: bytes, settings: Settings) -> IngestReport:
    """Index one uploaded PDF. Never raises for bad input; problems become status="error"."""
    # Cheap duplicate check on the raw bytes before parsing the whole PDF.
    file_hash = compute_file_hash(data) if data else ""
    try:
        if file_hash and store.has_file(file_hash):
            return IngestReport(
                filename=filename,
                status="duplicate",
                message="This file is already indexed; skipped re-indexing.",
                file_hash=file_hash,
            )
    except Exception as exc:
        logger.exception("Could not check for duplicates of %s", filename)
        return IngestReport(
            filename=filename,
            status="error",
            message=f"Unexpected error while checking the vector store ({type(exc).__name__}: {exc}).",
        )

    try:
        loaded = load_pdf_bytes(filename, data, max_file_mb=settings.max_file_mb)
    except PDFValidationError as exc:
        return IngestReport(filename=filename, status="error", message=str(exc))
    except Exception as exc:  # unexpected parser failure
        logger.exception("Unexpected error while reading %s", filename)
        return IngestReport(
            filename=filename,
            status="error",
            message=f"Unexpected error while reading the PDF ({type(exc).__name__}: {exc}).",
        )

    report = IngestReport(
        filename=filename,
        status="indexed",
        message="",
        pages=loaded.total_pages,
        empty_pages=list(loaded.empty_pages),
        file_hash=loaded.file_hash,
    )

    try:
        chunks = split_documents(loaded.pages, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            report.status = "error"
            report.message = "No text chunks could be produced from this PDF."
            return report
        report.chunks = store.add_chunks(chunks)
    except Exception as exc:
        logger.exception("Unexpected error while indexing %s", filename)
        report.status = "error"
        report.message = f"Unexpected error while indexing the PDF ({type(exc).__name__}: {exc})."
        return report

    parts = [f"Indexed {report.chunks} chunks from {report.pages} pages."]
    # The loader already describes skipped empty pages in its warnings; add our own only if it didn't.
    warnings = list(loaded.warnings)
    if report.empty_pages and not warnings:
        warnings.append(
            f"{len(report.empty_pages)} page(s) had no extractable text and were skipped: "
            + ", ".join(map(str, report.empty_pages))
        )
    parts.extend(f"Warning: {w}" for w in warnings)
    report.message = " ".join(parts)
    return report

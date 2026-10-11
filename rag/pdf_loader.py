"""PDF ingestion: validation, text extraction and normalization.

Every failure a user can trigger (wrong file type, corrupted or encrypted PDF,
image-only scan, ...) is reported as a `PDFValidationError` whose message is
safe to show directly in the UI.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import warnings as _warnings
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

with _warnings.catch_warnings():
    # langchain-community emits a deprecation notice on import; it is noise for end users.
    _warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.document_loaders import PyPDFLoader

logger = logging.getLogger(__name__)

PDF_MAGIC = b"%PDF-"
# Loader metadata keys worth keeping (all scalar); everything else is dropped.
_KEEP_LOADER_KEYS = ("total_pages", "page_label")


class PDFValidationError(Exception):
    """Raised for any PDF problem; str(e) is a user-facing explanation."""


@dataclass
class LoadResult:
    filename: str
    file_hash: str
    pages: list[Document]  # one Document per page that has text (normalized)
    total_pages: int
    empty_pages: list[int] = field(default_factory=list)  # 1-based
    warnings: list[str] = field(default_factory=list)


def compute_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_pdf_bytes(filename: str, data: bytes, max_file_mb: int = 50) -> None:
    """Cheap checks that do not require parsing the PDF."""
    name = os.path.basename(filename or "")
    if not name.lower().endswith(".pdf"):
        raise PDFValidationError(
            f"'{name or filename}' is not a PDF file. Only files with a .pdf extension are supported."
        )
    if not data:
        raise PDFValidationError(f"'{name}' is empty (0 bytes).")
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_file_mb:
        raise PDFValidationError(
            f"'{name}' is {size_mb:.1f} MB, which exceeds the {max_file_mb} MB upload limit."
        )
    if not data.lstrip()[:5].startswith(PDF_MAGIC):
        raise PDFValidationError(
            f"'{name}' does not look like a real PDF (missing %PDF- header). "
            "It may be another file type renamed to .pdf."
        )


_HYPHEN_BREAK = re.compile(r"(\w+)-[ \t]*\n[ \t]*(\w+)")
# Typographic ligatures (common in LaTeX PDFs) split words for the embedding tokenizer.
_LIGATURES = str.maketrans({"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl"})
_SPACES = re.compile(r"[ \t\f\v ]+")
_SPACE_AROUND_NL = re.compile(r"[ \t]*\n[ \t]*")
_MANY_NL = re.compile(r"\n{3,}")


def _rejoin_hyphen_break(match: re.Match, text: str) -> str:
    """Rejoin a word split across lines, keeping real compounds like "self-attention".

    Uses the page itself as a dictionary: if the hyphenated form appears elsewhere
    keep the hyphen; if the joined form appears, join. Otherwise join, unless the
    first part is capitalised or numeric ("English-to-German", "BERT-base").
    """
    left, right = match.group(1), match.group(2)
    hyphenated, joined = f"{left}-{right}", f"{left}{right}"
    if hyphenated in text:
        return hyphenated
    if joined in text:
        return joined
    if left[0].isupper() or any(ch.isdigit() for ch in left):
        return hyphenated
    return joined


def normalize_text(text: str) -> str:
    """Clean extracted PDF text while keeping paragraph structure and content."""
    if not text:
        return ""
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(_LIGATURES)  # "efﬁcient" -> "efficient"
    text = _HYPHEN_BREAK.sub(lambda m: _rejoin_hyphen_break(m, text), text)  # "embed-\nding" -> "embedding"
    text = _SPACES.sub(" ", text)
    text = _SPACE_AROUND_NL.sub("\n", text)
    text = _MANY_NL.sub("\n\n", text)
    return text.strip()


def _check_readable(name: str, path: str) -> int:
    """Open with pypdf to detect corruption/encryption early; returns page count."""
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            # PDFs with only an owner password open with an empty user password.
            try:
                ok = reader.decrypt("")
            except Exception:  # missing crypto backend, unsupported algorithm, ...
                ok = 0
            if not ok:
                raise PDFValidationError(
                    f"'{name}' is password-protected/encrypted. Please upload an unencrypted copy."
                )
        return len(reader.pages)
    except PDFValidationError:
        raise
    except Exception as exc:
        logger.info("Failed to open %s: %r", name, exc)
        raise PDFValidationError(
            f"'{name}' could not be read; the file appears to be corrupted or malformed."
        ) from exc


def load_pdf_bytes(filename: str, data: bytes, max_file_mb: int = 50) -> LoadResult:
    """Validate and extract normalized per-page Documents from raw PDF bytes."""
    validate_pdf_bytes(filename, data, max_file_mb)
    name = os.path.basename(filename)
    file_hash = compute_file_hash(data)

    # PyPDFLoader needs a path, so spill the bytes to a temp file.
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(data)
        tmp.close()
        total_pages = _check_readable(name, tmp.name)
        if total_pages == 0:
            raise PDFValidationError(f"'{name}' contains no pages.")
        try:
            raw_pages = PyPDFLoader(tmp.name).load()
        except Exception as exc:
            logger.info("Text extraction failed for %s: %r", name, exc)
            raise PDFValidationError(
                f"Text could not be extracted from '{name}'; the file may be corrupted."
            ) from exc
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    pages: list[Document] = []
    empty_pages: list[int] = []
    for i, doc in enumerate(raw_pages):
        page_no = int(doc.metadata.get("page", i)) + 1  # loader is 0-based
        text = normalize_text(doc.page_content)
        if not text:
            empty_pages.append(page_no)
            continue
        metadata = {"source": name, "page": page_no, "file_hash": file_hash}
        for key in _KEEP_LOADER_KEYS:
            value = doc.metadata.get(key)
            if isinstance(value, (str, int, float)) and not isinstance(value, bool):
                metadata[key] = value
        pages.append(Document(page_content=text, metadata=metadata))

    if not pages:
        raise PDFValidationError(
            f"No extractable text was found in '{name}'. It is probably a scanned or "
            "image-only PDF; OCR is not supported in this version."
        )

    warnings: list[str] = []
    if empty_pages:
        warnings.append(
            f"{len(empty_pages)} of {total_pages} page(s) had no extractable text and were skipped: "
            + ", ".join(map(str, empty_pages))
        )
    return LoadResult(name, file_hash, pages, total_pages, empty_pages, warnings)


def load_pdf_path(path: str | Path, max_file_mb: int = 50) -> LoadResult:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PDFValidationError(f"Could not read file '{path}': {exc.strerror or exc}") from exc
    return load_pdf_bytes(path.name, data, max_file_mb)

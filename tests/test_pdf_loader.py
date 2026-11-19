"""Tests for rag.pdf_loader: validation, extraction, metadata, normalization."""
from __future__ import annotations

import io

import pytest
from pypdf import PdfReader, PdfWriter

from rag.pdf_loader import (
    PDFValidationError,
    compute_file_hash,
    load_pdf_bytes,
    load_pdf_path,
    normalize_text,
    validate_pdf_bytes,
)
from tests.conftest import KNOWN_FACTS, make_pdf


# --- validation ---------------------------------------------------------------

def test_rejects_wrong_extension(sample_pdf_bytes):
    _, data = sample_pdf_bytes
    with pytest.raises(PDFValidationError, match="not a PDF"):
        validate_pdf_bytes("notes.txt", data)


def test_rejects_non_pdf_bytes():
    with pytest.raises(PDFValidationError, match="%PDF-"):
        load_pdf_bytes("fake.pdf", b"PK\x03\x04 this is a zip file")


def test_rejects_empty_file():
    with pytest.raises(PDFValidationError, match="empty"):
        load_pdf_bytes("empty.pdf", b"")


def test_rejects_oversized_file():
    data = b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024)
    with pytest.raises(PDFValidationError, match="exceeds the 1 MB"):
        load_pdf_bytes("big.pdf", data, max_file_mb=1)


def test_uppercase_extension_accepted(sample_pdf_bytes):
    _, data = sample_pdf_bytes
    validate_pdf_bytes("REPORT.PDF", data)  # no exception


def test_corrupted_pdf_gives_clear_error(corrupted_pdf_bytes):
    name, data = corrupted_pdf_bytes
    with pytest.raises(PDFValidationError, match="corrupted"):
        load_pdf_bytes(name, data)


def test_encrypted_pdf_gives_clear_error(sample_pdf_bytes):
    _, data = sample_pdf_bytes
    writer = PdfWriter(clone_from=PdfReader(io.BytesIO(data)))
    writer.encrypt(user_password="secret", algorithm="RC4-128")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(PDFValidationError, match="encrypted"):
        load_pdf_bytes("locked.pdf", buf.getvalue())


def test_blank_pdf_reports_scanned(blank_pdf_bytes):
    name, data = blank_pdf_bytes
    with pytest.raises(PDFValidationError, match="scanned") as exc:
        load_pdf_bytes(name, data)
    assert "OCR" in str(exc.value)


# --- extraction & metadata ----------------------------------------------------

def test_page_count_and_one_based_metadata(sample_pdf_bytes):
    name, data = sample_pdf_bytes
    result = load_pdf_bytes(name, data)
    assert result.filename == name
    assert result.total_pages == 3
    assert result.empty_pages == []
    assert [p.metadata["page"] for p in result.pages] == [1, 2, 3]
    for page in result.pages:
        assert page.metadata["source"] == name
        assert page.metadata["file_hash"] == compute_file_hash(data)
        # only scalar metadata survives (Chroma-safe)
        assert all(isinstance(v, (str, int, float)) for v in page.metadata.values())


def test_known_facts_on_expected_pages(sample_pdf_bytes):
    name, data = sample_pdf_bytes
    pages = {p.metadata["page"]: p.page_content for p in load_pdf_bytes(name, data).pages}
    for substring, page in KNOWN_FACTS.values():
        assert substring in pages[page]


def test_empty_pages_recorded_and_skipped():
    data = make_pdf(["First page has text.", None, "Third page has text too."])
    result = load_pdf_bytes("mixed.pdf", data)
    assert result.total_pages == 3
    assert result.empty_pages == [2]
    assert [p.metadata["page"] for p in result.pages] == [1, 3]
    assert result.warnings and "2" in result.warnings[0]


def test_source_is_basename(sample_pdf_bytes):
    _, data = sample_pdf_bytes
    result = load_pdf_bytes("/some/dir/paper.pdf", data)
    assert result.pages[0].metadata["source"] == "paper.pdf"


def test_load_pdf_path(sample_pdfs_dir, sample_pdf_bytes):
    name, _ = sample_pdf_bytes
    result = load_pdf_path(sample_pdfs_dir / name)
    assert result.total_pages == 3


def test_load_pdf_path_missing_file(tmp_path):
    with pytest.raises(PDFValidationError):
        load_pdf_path(tmp_path / "nope.pdf")


def test_hash_is_deterministic(sample_pdf_bytes, second_pdf_bytes):
    assert compute_file_hash(sample_pdf_bytes[1]) == compute_file_hash(sample_pdf_bytes[1])
    assert compute_file_hash(sample_pdf_bytes[1]) != compute_file_hash(second_pdf_bytes[1])


# --- normalization ------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("embed-\nding vectors", "embedding vectors"),
        ("too    many \t spaces", "too many spaces"),
        ("para one\n\n\n\n\npara two", "para one\n\npara two"),
        ("keep\n\nparagraphs", "keep\n\nparagraphs"),
        ("nul\x00l char", "null char"),
        ("  Trim Me  \n", "Trim Me"),
        ("Accuracy = 92.5% (p < 0.05) & F1: 0.88", "Accuracy = 92.5% (p < 0.05) & F1: 0.88"),
        ("well-known term", "well-known term"),
        ("line one   \n   line two", "line one\nline two"),
        ("", ""),
        # Ligatures from LaTeX PDFs are expanded so the tokenizer sees whole words.
        ("e\ufb03cient \ufb02exible \ufb01xed", "efficient flexible fixed"),
        # Capitalised first part = real compound, keep the hyphen.
        ("WMT 2014 English-\nto-German task", "WMT 2014 English-to-German task"),
        # Hyphenated form used elsewhere on the page = keep the hyphen.
        ("self-attention layers use self-\nattention", "self-attention layers use self-attention"),
    ],
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_preserves_case():
    assert normalize_text("BERT and GPT") == "BERT and GPT"


# --- running header / footer stripping (V1.3) ---------------------------------------

def test_strip_repeated_lines_removes_running_header_and_page_numbers():
    from rag.pdf_loader import strip_repeated_lines

    pages = [
        f"ACME Annual Report 2024\nSection {i} body text about topic {i}.\nMore unique content {i}.\n{i}"
        for i in range(1, 7)
    ]
    out = strip_repeated_lines(pages)
    # kept once (first page) so the title/date stays searchable, removed elsewhere
    assert out[0].startswith("ACME Annual Report 2024")
    assert all("ACME Annual Report" not in p for p in out[1:])
    # bare page numbers removed after their first occurrence
    assert all(not p.endswith(f"\n{i}") for i, p in enumerate(out[1:], start=2))
    # body text untouched
    assert all(f"Section {i} body text" in out[i - 1] for i in range(1, 7))


def test_strip_repeated_lines_keeps_short_docs_and_body_repeats():
    from rag.pdf_loader import strip_repeated_lines

    short = ["Header\nA", "Header\nB", "Header\nC"]
    assert strip_repeated_lines(short) == short  # fewer than 4 pages: unchanged
    # a repeated line in the middle of pages (not at an edge) is content, not a header
    pages = [f"Top {i}\nl2 {i}\nl3 {i}\nl4 {i}\nRepeated body line\nl6 {i}\nl7 {i}\nl8 {i}\nend {i}" for i in range(6)]
    assert all("Repeated body line" in p for p in strip_repeated_lines(pages))

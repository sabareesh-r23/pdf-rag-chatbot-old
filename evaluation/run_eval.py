"""Batch evaluation runner for the RAG PDF chatbot.

Ingests every PDF in ``data/sample_pdfs/`` into a dedicated evaluation
collection (named after the chunking parameters so different configurations
never mix), asks every question from a questions file through
``RAGPipeline.ask`` and writes a JSON + Markdown record of what happened.

Only *observable* facts are recorded automatically (retrieved chunks, answer,
sources, abstention flag, latency and whether an expected page was retrieved).
Human judgement fields (correct / relevant / supported / abstention_correct /
failure_class / observations) are deliberately left blank for the evaluator.

Usage:
    python -m evaluation.run_eval --questions evaluation/questions.json
    python -m evaluation.run_eval --top-k 6 --chunk-size 500 --chunk-overlap 100 --reindex
    python -m evaluation.run_eval --retrieval-mode mmr --mmr-lambda 0.5 --mmr-fetch-k 20 --no-rerank
    python -m evaluation.run_eval --retrieval-mode similarity --no-rerank     # V1.1/V1.2 behaviour
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from rag.config import FALLBACK_ANSWER, PROJECT_ROOT, Settings, get_settings

DEFAULT_PDF_DIR = PROJECT_ROOT / "data" / "sample_pdfs"
DEFAULT_QUESTIONS = PROJECT_ROOT / "evaluation" / "questions.json"
DEFAULT_OUT = PROJECT_ROOT / "evaluation" / "results"
DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
VALID_CATEGORIES = {"factual", "multi_chunk", "synthesis", "unanswerable", "adversarial"}

# Filled in by the human evaluator; never auto-populated.
HUMAN_FIELDS = ("correct", "relevant", "supported", "abstention_correct", "failure_class", "observations")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def eval_collection_name(settings: Settings, pdf_dir: Path | None = None) -> str:
    """Collection name derived from the parameters that change the index contents."""
    name = f"eval_cs{settings.chunk_size}_co{settings.chunk_overlap}"
    if pdf_dir is not None and pdf_dir.resolve() != DEFAULT_PDF_DIR.resolve():
        # Separate corpora (e.g. data/long_pdfs) get their own collection.
        name += "_" + re.sub(r"[^a-zA-Z0-9]+", "_", pdf_dir.name)[:24]
    if settings.embedding_model != DEFAULT_EMBEDDING:
        name += "_emb" + hashlib.sha1(settings.embedding_model.encode()).hexdigest()[:8]
    return name


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        questions = json.load(fh)
    if not isinstance(questions, list):
        raise ValueError(f"{path} must contain a JSON list of question objects")
    seen: set[str] = set()
    for q in questions:
        for key in ("id", "category", "question"):
            if key not in q:
                raise ValueError(f"Question {q!r} is missing required key {key!r}")
        if q["category"] not in VALID_CATEGORIES:
            raise ValueError(f"Question {q['id']}: unknown category {q['category']!r}")
        if q["id"] in seen:
            raise ValueError(f"Duplicate question id {q['id']!r}")
        seen.add(q["id"])
        q.setdefault("expected_answer", "")
        q.setdefault("expected_sources", [])
        q.setdefault("notes", "")
        q.setdefault("evidence_snippets", [])
    return questions


def expected_page_retrieved(expected_sources: list[dict], chunks: list[dict]) -> bool | None:
    """True if any expected (source, page) pair is among the retrieved chunks.

    Returns None when the question has no expected evidence (e.g. unanswerable).
    """
    expected = {(s["source"], int(p)) for s in expected_sources for p in s.get("pages", [])}
    if not expected:
        return None
    retrieved = {(c["source"], int(c["page"])) for c in chunks}
    return bool(expected & retrieved)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_retrieved(snippets: list[str], chunks: list[dict]) -> str | None:
    """Stricter hint than the page check: are the verbatim evidence snippets in the retrieved text?

    Returns "all", "partial" or "none", or None when the question lists no snippets.
    """
    if not snippets:
        return None
    joined = " ".join(_squash(c["text"]) for c in chunks)
    found = sum(1 for snip in snippets if _squash(snip) in joined)
    return "all" if found == len(snippets) else ("partial" if found else "none")


def ingest_corpus(store, settings: Settings, pdf_dir: Path) -> tuple[list[dict], list[str]]:
    """Ingest all PDFs in pdf_dir; returns (ingest report dicts, file hashes of usable files)."""
    from rag.ingest import ingest_file

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")
    reports, hashes = [], []
    for pdf in pdfs:
        t0 = time.perf_counter()
        report = ingest_file(store, pdf.name, pdf.read_bytes(), settings)
        elapsed = time.perf_counter() - t0
        print(f"  [{report.status:9}] {pdf.name}: {report.message} ({elapsed:.1f}s)")
        reports.append(
            {
                "filename": report.filename,
                "status": report.status,
                "message": report.message,
                "pages": report.pages,
                "empty_pages": report.empty_pages,
                "chunks": report.chunks,
                "file_hash": report.file_hash,
                "seconds": round(elapsed, 2),
            }
        )
        if report.status in ("indexed", "duplicate") and report.file_hash:
            hashes.append(report.file_hash)
    return reports, hashes


def run_question(pipeline, q: dict, file_hashes: list[str], top_k: int) -> dict:
    """Ask one question and capture everything observable about the answer."""
    record = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "expected_answer": q.get("expected_answer", ""),
        "expected_sources": q.get("expected_sources", []),
        "notes": q.get("notes", ""),
    }
    t0 = time.perf_counter()
    try:
        response = pipeline.ask(q["question"], file_hashes=file_hashes, top_k=top_k)
        error = None
    except Exception as exc:  # recorded, the run continues (LLM init errors abort in main)
        response, error = None, f"{type(exc).__name__}: {exc}"
        if type(exc).__name__ == "LLMInitError":
            raise
    latency = time.perf_counter() - t0

    chunks = []
    if response is not None:
        chunks = [
            {
                "rank": i,
                "source": c.source,
                "page": c.page,
                "score": round(float(c.score), 4),
                "rerank_score": None if c.rerank_score is None else round(float(c.rerank_score), 3),
                "chunk_id": c.chunk_id,
                "text": c.text,
            }
            for i, c in enumerate(response.chunks, start=1)
        ]
    record.update(
        {
            "retrieved_chunks": chunks,
            "sub_queries": list(getattr(response, "sub_queries", []) or []) if response is not None else [],
            "actual_answer": response.answer if response is not None else "",
            "sources": list(response.sources) if response is not None else [],
            "abstained": bool(response.abstained) if response is not None else None,
            "latency_seconds": round(latency, 2),
            "error": error,
            # Automatic hint only - NOT a judgement of correctness.
            "expected_page_retrieved": expected_page_retrieved(q.get("expected_sources", []), chunks),
            "evidence_retrieved": evidence_retrieved(q.get("evidence_snippets", []), chunks),
            "human_judgement": {field: "" for field in HUMAN_FIELDS},
        }
    )
    return record


# --------------------------------------------------------------------------- #
# Markdown rendering
# --------------------------------------------------------------------------- #
def _cell(text: object, limit: int | None = None) -> str:
    s = "" if text is None else str(text)
    s = re.sub(r"\s+", " ", s).strip()
    if limit and len(s) > limit:
        s = s[: limit - 1] + "…"
    return s.replace("|", "\\|")


def _fmt_expected(sources: list[dict]) -> str:
    if not sources:
        return "— (none: not in corpus)"
    return "; ".join(f"{s['source']} p. {', '.join(map(str, s.get('pages', [])))}" for s in sources)


def _hint(value: bool | None) -> str:
    return {True: "Y", False: "N", None: "N/A"}[value]


def render_markdown(result: dict) -> str:
    cfg = result["config"]
    lines = [
        f"# Evaluation run {result['run_id']}",
        "",
        "> Auto-generated by `evaluation/run_eval.py`. The *Human judgement* fields are intentionally",
        "> blank — judged in `docs/evaluation_report.md` after reading each answer",
        "> against the retrieved evidence. `expected_page_retrieved` is only an automatic hint.",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for key in (
        "llm_provider", "llm_model", "llm_temperature", "llm_top_p", "llm_top_k", "llm_num_ctx", "embedding_model",
        "chunk_size", "chunk_overlap", "retrieval_mode", "top_k", "mmr_fetch_k", "mmr_lambda", "rerank",
        "rerank_candidates", "query_decomposition",
        "min_relevance", "collection_name", "chroma_dir",
    ):
        lines.append(f"| {key} | `{cfg.get(key)}` |")
    lines += [
        f"| questions file | `{result['questions_file']}` |",
        f"| started | {result['started_at']} |",
        f"| python / platform | {result['environment']['python']} / {result['environment']['platform']} |",
        "",
        "## Corpus ingestion",
        "",
        "| File | Status | Pages | Chunks | Message |",
        "|---|---|---|---|---|",
    ]
    for r in result["ingestion"]:
        lines.append(
            f"| {_cell(r['filename'])} | {r['status']} | {r['pages']} | {r['chunks']} | {_cell(r['message'], 160)} |"
        )

    qs = result["questions"]
    lat = [q["latency_seconds"] for q in qs if q.get("error") is None]
    lines += [
        "",
        "## Automatic summary (observations only, no correctness judgement)",
        "",
        f"- Questions run: {len(qs)}; errors: {sum(1 for q in qs if q.get('error'))}",
        f"- Abstained (fallback answer): {sum(1 for q in qs if q.get('abstained'))}",
        f"- Expected page retrieved (answerable questions): "
        f"{sum(1 for q in qs if q['expected_page_retrieved'] is True)}"
        f" / {sum(1 for q in qs if q['expected_page_retrieved'] is not None)}",
        f"- Evidence snippets retrieved (all / partial / none): "
        f"{sum(1 for q in qs if q.get('evidence_retrieved') == 'all')} / "
        f"{sum(1 for q in qs if q.get('evidence_retrieved') == 'partial')} / "
        f"{sum(1 for q in qs if q.get('evidence_retrieved') == 'none')}",
        f"- Mean latency: {sum(lat) / len(lat):.2f}s" if lat else "- Mean latency: n/a",
        "",
        "## Overview",
        "",
        "| ID | Category | Expected page retrieved? | Evidence retrieved? | Abstained | Latency (s) | Correct? | Relevant? | Supported? | Correct abstention? | Failure class |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for q in qs:
        lines.append(
            f"| {q['id']} | {q['category']} | {_hint(q['expected_page_retrieved'])} | "
            f"{q.get('evidence_retrieved') or 'N/A'} | "
            f"{'Y' if q['abstained'] else ('N' if q['abstained'] is False else 'error')} | "
            f"{q['latency_seconds']} |  |  |  |  |  |"
        )

    lines += ["", "## Per-question detail", ""]
    for q in qs:
        lines += [
            f"### {q['id']} ({q['category']})",
            "",
            f"**Question:** {q['question']}",
            "",
            f"**Expected answer (verified):** {q['expected_answer']}",
            "",
            f"**Expected page(s):** {_fmt_expected(q['expected_sources'])}",
            "",
            f"**Ground-truth notes:** {q['notes']}",
            "",
        ]
        if q.get("error"):
            lines += [f"**ERROR:** `{q['error']}`", ""]
        lines += [
            "**Retrieved chunks:**",
            "",
            "| Rank | Source | Page | Score | Text (truncated) |",
            "|---|---|---|---|---|",
        ]
        for c in q["retrieved_chunks"]:
            lines.append(
                f"| {c['rank']} | {_cell(c['source'])} | {c['page']} | {c['score']:.4f} | {_cell(c['text'], 300)} |"
            )
        if not q["retrieved_chunks"]:
            lines.append("| — | — | — | — | (no chunks retrieved) |")
        sources = ", ".join(f"{s.get('source')} p. {s.get('page')}" for s in q["sources"]) or "—"
        lines += [
            "",
            "**Actual answer:**",
            "",
            "> " + (q["actual_answer"] or "").replace("\n", "\n> "),
            "",
            f"**Sources shown:** {sources}  ",
            f"**Abstained:** {q['abstained']}  ",
            f"**Latency:** {q['latency_seconds']} s  ",
            f"**Expected page retrieved (auto hint):** {_hint(q['expected_page_retrieved'])}  ",
            f"**Evidence snippets retrieved (auto hint):** {q.get('evidence_retrieved') or 'N/A'}",
            "",
            "**Human judgement (fill in):**",
            "",
            "| Correct? | Relevant? | Supported by context? | Correct abstention? | Failure class | Observations & improvement |",
            "|---|---|---|---|---|---|",
            "|  |  |  |  |  |  |",
            "",
        ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m evaluation.run_eval",
        description="Run the RAG pipeline over an evaluation question set and record results.",
    )
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="questions JSON file")
    p.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="directory of PDFs to index")
    p.add_argument("--top-k", type=int, default=None, help="chunks to retrieve (default: TOP_K env / 4)")
    p.add_argument("--chunk-size", type=int, default=None, help="chunk size in characters (default: CHUNK_SIZE / 1000)")
    p.add_argument("--chunk-overlap", type=int, default=None, help="chunk overlap (default: CHUNK_OVERLAP / 200)")
    p.add_argument("--retrieval-mode", choices=["similarity", "mmr", "hybrid"], default=None,
                   help="hybrid, similarity or mmr (default: RETRIEVAL_MODE / hybrid)")
    p.add_argument("--mmr-fetch-k", type=int, default=None, help="MMR candidate pool (default: MMR_FETCH_K / 20)")
    p.add_argument("--mmr-lambda", type=float, default=None, help="MMR relevance weight 0-1 (default: MMR_LAMBDA / 0.5)")
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                   help="--rerank / --no-rerank: cross-encoder reranking (default: RERANK / on)")
    p.add_argument("--decompose", action=argparse.BooleanOptionalAction, default=None,
                   help="--decompose / --no-decompose: split comparison questions into sub-queries (default: QUERY_DECOMPOSITION / off)")
    p.add_argument("--temperature", type=float, default=None, help="override LLM_TEMPERATURE")
    p.add_argument("--top-p", type=float, default=None, help="override LLM_TOP_P")
    p.add_argument("--llm-top-k", type=int, default=None, help="override LLM_TOP_K (sampling, not retrieval)")
    p.add_argument("--num-ctx", type=int, default=None, help="override LLM_NUM_CTX (Ollama context window)")
    p.add_argument("--llm-model", default=None, help="override LLM_MODEL (for model comparisons)")
    p.add_argument("--llm-provider", choices=["ollama", "openai"], default=None, help="override LLM_PROVIDER")
    p.add_argument("--ids", default=None, help="comma-separated question ids to run (default: all)")
    p.add_argument("--limit", type=int, default=None, help="run only the first N questions (smoke test)")
    p.add_argument("--reindex", action="store_true", help="wipe and rebuild the evaluation collection")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory for results files")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    questions = load_questions(args.questions)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",") if i.strip()}
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit is not None:
        questions = questions[: args.limit]
    if not questions:
        print("No questions selected.", file=sys.stderr)
        return 2

    base = get_settings(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        llm_model=args.llm_model,
        llm_provider=args.llm_provider,
        retrieval_mode=args.retrieval_mode,
        mmr_fetch_k=args.mmr_fetch_k,
        mmr_lambda=args.mmr_lambda,
        rerank=args.rerank,
        query_decomposition=args.decompose,
        llm_temperature=args.temperature,
        llm_top_p=args.top_p,
        llm_num_ctx=args.num_ctx,
        llm_top_k=args.llm_top_k,
    )
    settings = get_settings(**{**base.as_report_dict(), "collection_name": eval_collection_name(base, args.pdf_dir)})

    # Imported lazily so `--help` works even before heavy deps / all modules exist.
    from rag.generation import LLMInitError
    from rag.pipeline import RAGPipeline
    from rag.vectorstore import VectorStoreManager

    print(f"Collection: {settings.collection_name}  (chroma_dir={settings.chroma_dir})")
    print("Loading embedding model and vector store...")
    store = VectorStoreManager(settings)
    if args.reindex:
        print("Re-indexing: resetting evaluation collection.")
        store.reset()
    print(f"Ingesting PDFs from {args.pdf_dir}:")
    ingestion, file_hashes = ingest_corpus(store, settings, args.pdf_dir)
    if not file_hashes:
        print("No PDFs could be indexed; aborting.", file=sys.stderr)
        return 1

    pipeline = RAGPipeline(settings, store)
    started = datetime.now()
    run_id = started.strftime("%Y%m%d_%H%M%S")
    records = []
    for n, q in enumerate(questions, start=1):
        print(f"[{n}/{len(questions)}] {q['id']} ({q['category']}): {q['question']}")
        try:
            rec = run_question(pipeline, q, file_hashes, settings.top_k)
        except LLMInitError as exc:
            print(f"\nLLM unavailable — aborting run: {exc}", file=sys.stderr)
            return 1
        records.append(rec)
        status = "ERROR" if rec["error"] else ("abstained" if rec["abstained"] else "answered")
        print(
            f"    -> {status}; expected page retrieved: {_hint(rec['expected_page_retrieved'])}; "
            f"evidence: {rec['evidence_retrieved'] or 'N/A'}; "
            f"{rec['latency_seconds']}s"
        )

    result = {
        "run_id": run_id,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "questions_file": str(args.questions),
        "pdf_dir": str(args.pdf_dir),
        "config": settings.as_report_dict(),
        "fallback_answer": FALLBACK_ANSWER,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "ingestion": ingestion,
        "human_fields": list(HUMAN_FIELDS),
        "questions": records,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"results_{run_id}.json"
    md_path = args.out / f"results_{run_id}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

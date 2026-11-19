"""Streamlit UI for the PDF RAG chatbot.

Run with:  streamlit run app.py

The UI only uses the public APIs described in CONTRACTS.md:
ingestion (rag.ingest), the vector store (rag.vectorstore) and the
RAG pipeline (rag.pipeline). Heavy objects (embedding model, Chroma client,
pipeline/LLM) are cached with st.cache_resource.
"""
from __future__ import annotations

import hashlib

import streamlit as st

from rag.config import Settings, get_settings
from rag.embeddings import EmbeddingInitError, get_embeddings
from rag.generation import LLMInitError
from rag.ingest import ingest_file
from rag.pipeline import RAGPipeline
from rag.reranker import RerankerInitError, get_reranker
from rag.vectorstore import VectorStoreManager

try:  # optional helper (not in CONTRACTS.md); degrade gracefully if absent
    from rag.generation import check_ollama
except ImportError:  # pragma: no cover
    check_ollama = None

st.set_page_config(page_title="PDF RAG Chatbot", page_icon=":page_facing_up:", layout="wide")

APP_ERRORS = (EmbeddingInitError, LLMInitError, RerankerInitError, ValueError)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "messages": [],          # chat history: list of dicts
        "uploaded_bytes": {},    # file_hash -> (filename, bytes) for re-indexing
        "ingest_reports": [],    # last ingest run results (list of dicts)
        "deselected": set(),     # file_hashes the user unticked (default: all selected)
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# --------------------------------------------------------------------------- #
# Cached heavy resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_store(chroma_dir: str, collection_name: str, embedding_model: str) -> VectorStoreManager:
    # Keyed only by the fields that affect the store so top-k changes don't reload.
    settings = get_settings(
        chroma_dir=chroma_dir, collection_name=collection_name, embedding_model=embedding_model
    )
    embeddings = get_embeddings(embedding_model)
    return VectorStoreManager(settings, embeddings=embeddings)


@st.cache_resource(show_spinner=False)
def get_pipeline(settings: Settings) -> RAGPipeline:
    # LLM is created lazily by the pipeline on first ask().
    return RAGPipeline(settings, get_store(settings.chroma_dir, settings.collection_name, settings.embedding_model))


@st.cache_data(ttl=30, show_spinner=False)
def cached_ollama_status(settings: Settings):
    return check_ollama(settings)


def normalize_status(result) -> tuple[bool, str]:
    """check_ollama's return shape is not pinned in CONTRACTS.md; accept common shapes."""
    if isinstance(result, tuple) and len(result) >= 2:
        return bool(result[0]), str(result[1])
    if isinstance(result, dict):
        ok = result.get("ok", result.get("ready", False))
        return bool(ok), str(result.get("message", ""))
    if isinstance(result, bool):
        return result, ""
    if hasattr(result, "ok") or hasattr(result, "ready"):
        ok = getattr(result, "ok", getattr(result, "ready", False))
        return bool(ok), str(getattr(result, "message", ""))
    if isinstance(result, str):  # a message string usually signals a problem
        return False, result
    return bool(result), ""


# --------------------------------------------------------------------------- #
# Settings (top-k / chunking overrides for this session)
# --------------------------------------------------------------------------- #
try:
    base_settings = get_settings()
except ValueError as exc:
    st.error(f"Invalid configuration: {exc}")
    st.stop()

st.title("PDF RAG Chatbot")
st.caption("Ask questions answered only from the PDFs you upload. Answers cite file and page.")

with st.sidebar:
    with st.expander("Settings", expanded=False):
        st.markdown("**Retrieval**")
        top_k = st.slider("Top-k retrieved chunks", 1, 10, value=min(max(base_settings.top_k, 1), 10))
        modes = ["similarity", "mmr", "hybrid"]
        retrieval_mode = st.radio(
            "Search type", modes, index=modes.index(base_settings.retrieval_mode), horizontal=True,
            format_func=lambda m: {"similarity": "Similarity", "mmr": "MMR (diverse)", "hybrid": "Hybrid (keyword + meaning)"}[m],
            help="Similarity returns the k closest chunks by meaning. MMR (Maximal Marginal Relevance) first "
                 "fetches more candidates, then picks k that are relevant but not near-duplicates. Hybrid "
                 "combines a keyword (BM25) ranking with the meaning ranking, so exact terms are not missed.",
        )
        mmr_fetch_k, mmr_lambda = base_settings.mmr_fetch_k, base_settings.mmr_lambda
        if retrieval_mode == "mmr":
            mmr_fetch_k = st.slider("MMR candidates (fetch_k)", 5, 50, value=min(max(mmr_fetch_k, 5), 50))
            mmr_lambda = st.slider(
                "MMR lambda (1 = relevance, 0 = diversity)", 0.0, 1.0, value=float(mmr_lambda), step=0.1
            )
        rerank = st.checkbox(
            "Rerank with cross-encoder", value=base_settings.rerank,
            help=f"Retrieve {base_settings.rerank_candidates} candidates, rescore each against the question with "
                 f"{base_settings.rerank_model}, keep the best top-k. Slower, usually more accurate.",
        )
        st.markdown("**LLM**")
        llm_temperature = st.slider(
            "Temperature", 0.0, 1.0, value=min(float(base_settings.llm_temperature), 1.0), step=0.1,
            help="0 = deterministic, most faithful to the context. Higher values add variety and risk.",
        )
        llm_top_p = st.slider(
            "Top-p (nucleus sampling)", 0.1, 1.0, value=float(base_settings.llm_top_p), step=0.05,
            help="Only has an effect when temperature > 0.",
        )
        llm_top_k = st.slider(
            "Top-k sampling (LLM tokens)", 1, 100, value=min(max(int(base_settings.llm_top_k), 1), 100),
            help="Sample only from the K most likely next tokens. Only has an effect when temperature > 0. "
                 "Not the same as 'Top-k retrieved chunks'.",
        )
        ctx_options = [2048, 4096, 8192, 16384]
        llm_num_ctx = st.select_slider(
            "Context window (tokens, Ollama)", ctx_options,
            value=base_settings.llm_num_ctx if base_settings.llm_num_ctx in ctx_options else 4096,
            help="Prompt + answer must fit. Larger windows allow higher top-k but use more memory.",
        )
        st.markdown("**Chunking**")
        chunk_size = st.number_input(
            "Chunk size (characters)", min_value=100, max_value=8000, step=100,
            value=base_settings.chunk_size,
        )
        chunk_overlap = st.number_input(
            "Chunk overlap (characters)", min_value=0, max_value=4000, step=50,
            value=base_settings.chunk_overlap,
        )
        st.caption(
            "Chunk size/overlap apply to newly indexed files. To apply them to the existing "
            "corpus use 'Re-index with current settings' below."
        )

try:
    settings = get_settings(
        top_k=int(top_k), chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap),
        retrieval_mode=retrieval_mode, mmr_fetch_k=int(mmr_fetch_k), mmr_lambda=float(mmr_lambda), rerank=bool(rerank),
        llm_temperature=float(llm_temperature), llm_top_p=float(llm_top_p), llm_top_k=int(llm_top_k), llm_num_ctx=int(llm_num_ctx),
    )
except ValueError as exc:
    st.sidebar.error(f"Invalid settings: {exc}")
    settings = base_settings

with st.sidebar.expander("Effective configuration", expanded=False):
    st.markdown(
        f"- **Chunk size / overlap:** {settings.chunk_size} / {settings.chunk_overlap}\n"
        f"- **Embedding model:** `{settings.embedding_model}`\n"
        f"- **Retrieval:** {settings.retrieval_mode}, top-k {settings.top_k}"
        + (f" (MMR fetch_k {settings.mmr_fetch_k}, lambda {settings.mmr_lambda})" if settings.retrieval_mode == "mmr" else "")
        + (f", reranked from {settings.rerank_candidates} candidates" if settings.rerank else "")
        + "\n"
        f"- **Min relevance:** {settings.min_relevance}\n"
        f"- **LLM:** {settings.llm_provider} / `{settings.llm_model}` (temperature {settings.llm_temperature}, top-p {settings.llm_top_p}, sampling top-k {settings.llm_top_k}, context {settings.llm_num_ctx} tokens)\n"
        f"- **Vector store:** `{settings.chroma_dir}` (collection `{settings.collection_name}`)"
    )

# --------------------------------------------------------------------------- #
# Load the store (embedding model) - fatal for the app if it fails
# --------------------------------------------------------------------------- #
try:
    store = get_store(settings.chroma_dir, settings.collection_name, settings.embedding_model)
except APP_ERRORS as exc:
    st.error(f"Could not initialise the embedding model / vector store: {exc}")
    st.stop()
except Exception as exc:  # noqa: BLE001 - show rather than crash
    st.error(f"Unexpected error while opening the vector store: {exc}")
    st.stop()

# Load the reranker up front (cached per process) so the first question is not slowed
# down by the model load. A failure here is not fatal: answers can still be produced
# with reranking switched off in Settings.
if settings.rerank:
    try:
        with st.spinner("Loading reranker model..."):
            get_reranker(settings.rerank_model)
    except RerankerInitError as exc:
        st.sidebar.warning(f"{exc} Turn off 'Rerank with cross-encoder' in Settings to continue without it.")


def run_ingest(files: list[tuple[str, bytes]], ingest_settings: Settings, label: str) -> list[dict]:
    reports: list[dict] = []
    with st.status(label, expanded=True) as status:
        progress = st.progress(0.0)
        for i, (name, data) in enumerate(files, start=1):
            st.write(f"Processing **{name}** ...")
            try:
                rep = ingest_file(store, name, data, ingest_settings)
                report = {
                    "filename": rep.filename, "status": rep.status, "message": rep.message,
                    "pages": rep.pages, "empty_pages": list(rep.empty_pages), "chunks": rep.chunks,
                    "file_hash": rep.file_hash,
                }
            except Exception as exc:  # noqa: BLE001 - ingest should not raise, but be safe
                report = {"filename": name, "status": "error", "message": str(exc), "pages": 0,
                          "empty_pages": [], "chunks": 0, "file_hash": ""}
            if report["status"] in ("indexed", "duplicate"):
                fh = report["file_hash"] or hashlib.sha256(data).hexdigest()
                st.session_state.uploaded_bytes[fh] = (name, data)
            reports.append(report)
            progress.progress(i / len(files))
        n_err = sum(r["status"] == "error" for r in reports)
        status.update(
            label=f"{label}: done ({len(reports) - n_err} ok, {n_err} error)",
            state="error" if n_err == len(reports) else "complete",
            expanded=n_err > 0,
        )
    return reports


def render_report(r: dict) -> None:
    detail = f"pages: {r['pages']}, chunks: {r['chunks']}"
    if r["empty_pages"]:
        detail += f", empty pages (no text): {r['empty_pages']}"
    msg = f"**{r['filename']}** - {r['status']}: {r['message']} ({detail})"
    if r["status"] == "indexed":
        st.success(msg)
    elif r["status"] == "duplicate":
        st.info(msg)
    else:
        st.error(f"**{r['filename']}** - error: {r['message']}")


# --------------------------------------------------------------------------- #
# Sidebar: upload + index
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("1. Upload PDFs")
    uploads = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True,
        help="Files are validated server-side (renamed non-PDFs are rejected).",
    )
    if st.button("Index uploaded PDFs", type="primary", disabled=not uploads):
        files = [(u.name, u.getvalue()) for u in uploads]
        st.session_state.ingest_reports = run_ingest(files, settings, "Indexing PDFs")
        st.rerun()

    if st.session_state.ingest_reports:
        with st.expander("Last indexing results", expanded=True):
            for r in st.session_state.ingest_reports:
                render_report(r)

def clear_corpus() -> None:
    """Callback (runs before the rerun, so widget state may be modified here)."""
    try:
        store.reset()
        st.session_state.messages = []
        st.session_state.uploaded_bytes = {}
        st.session_state.ingest_reports = []
        st.session_state.deselected = set()
        st.session_state.confirm_clear = False
        st.toast("Corpus cleared")
    except Exception as exc:  # noqa: BLE001
        st.session_state.clear_error = f"Could not clear corpus: {exc}"


# --------------------------------------------------------------------------- #
# Sidebar: active corpus
# --------------------------------------------------------------------------- #
try:
    indexed_files = store.list_files()
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"Could not read the vector store: {exc}")
    indexed_files = []

with st.sidebar:
    st.header("2. Active corpus")
    selected_hashes: list[str] = []
    if not indexed_files:
        st.info("No documents indexed yet.")
    else:
        st.caption("Only ticked files are searched when answering.")
        for f in indexed_files:
            fh = f["file_hash"]
            col1, col2 = st.columns([5, 1])
            with col1:
                checked = st.checkbox(
                    f"{f['source']} ({f['chunks']} chunks)",
                    value=fh not in st.session_state.deselected,
                    key=f"sel_{fh}",
                )
            with col2:
                if st.button("X", key=f"rm_{fh}", help=f"Remove {f['source']} from the index"):
                    try:
                        store.remove_file(fh)
                        st.session_state.uploaded_bytes.pop(fh, None)
                        st.session_state.deselected.discard(fh)
                        st.toast(f"Removed {f['source']}")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Could not remove {f['source']}: {exc}")
                    st.rerun()
            if checked:
                selected_hashes.append(fh)
                st.session_state.deselected.discard(fh)
            else:
                st.session_state.deselected.add(fh)
        total_chunks = sum(f["chunks"] for f in indexed_files)
        st.caption(f"{len(selected_hashes)}/{len(indexed_files)} files selected, {total_chunks} chunks indexed.")

        st.divider()
        confirm = st.checkbox("I understand this deletes the whole index", key="confirm_clear")
        st.button("Clear corpus", disabled=not confirm, on_click=clear_corpus)
        if err := st.session_state.pop("clear_error", None):
            st.error(err)

        # Re-index: only possible for files whose bytes we hold in this session.
        held = st.session_state.uploaded_bytes
        indexed_hashes = {f["file_hash"] for f in indexed_files}
        missing = [f["source"] for f in indexed_files if f["file_hash"] not in held]
        with st.expander("Re-index with current settings"):
            st.caption(
                f"Resets the index and re-ingests uploaded files with chunk size "
                f"{settings.chunk_size} / overlap {settings.chunk_overlap}."
            )
            if missing:
                st.warning(
                    "These files were indexed in an earlier session and their bytes are not "
                    f"available, so they would be dropped: {', '.join(missing)}"
                )
            if st.button("Re-index now", disabled=not (held.keys() & indexed_hashes)):
                files = [held[h] for h in held if h in indexed_hashes]
                try:
                    store.reset()
                    st.session_state.messages = []
                    st.session_state.ingest_reports = run_ingest(files, settings, "Re-indexing")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Re-index failed: {exc}")
                st.rerun()

# --------------------------------------------------------------------------- #
# LLM status
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("3. LLM status")
    if settings.llm_provider == "ollama":
        if check_ollama is None:
            st.caption(f"Ollama / {settings.llm_model} (status check unavailable)")
        else:
            try:
                ok, message = normalize_status(cached_ollama_status(settings))
            except Exception as exc:  # noqa: BLE001
                ok, message = False, str(exc)
            if ok:
                st.success(f"Ollama ready: {settings.llm_model}" + (f" - {message}" if message else ""))
            else:
                st.error(message or f"Ollama is not ready for model {settings.llm_model}.")
            if st.button("Re-check LLM"):
                cached_ollama_status.clear()
                st.rerun()
    else:
        st.info(f"Provider: {settings.llm_provider} / {settings.llm_model} (OPENAI_API_KEY from environment)")

# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
def render_assistant(msg: dict) -> None:
    if msg.get("error"):
        st.error(msg["error"])
        return
    st.markdown(msg["answer"])
    if not msg.get("abstained") and msg.get("sources"):
        src = "; ".join(f"{s['source']} (p. {s['page']})" for s in msg["sources"])
        st.caption(f"**Sources:** {src}")
    chunks = msg.get("chunks", [])
    with st.expander(f"Retrieved evidence ({len(chunks)} chunks, top-k={msg.get('top_k')}, {msg.get('mode', 'similarity')})"):
        if not chunks:
            st.write("No chunks retrieved.")
        for i, c in enumerate(chunks, start=1):
            st.markdown(
                f"**#{i}** - `{c['source']}` page {c['page']} - similarity **{c['score']:.3f}** "
                + (f"- rerank **{c['rerank']:.2f}** " if c.get("rerank") is not None else "")
                + f"- chunk_id `{c['chunk_id']}`"
            )
            st.text(c["text"])
            st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

corpus_empty = not indexed_files
nothing_selected = bool(indexed_files) and not selected_hashes
if corpus_empty:
    st.info("Upload and index at least one PDF in the sidebar to start chatting.")
elif nothing_selected:
    st.warning("No files are selected in the active corpus. Tick at least one file in the sidebar.")

question = st.chat_input(
    "Ask a question about your documents", disabled=corpus_empty or nothing_selected
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        entry: dict = {"role": "assistant", "top_k": settings.top_k,
                       "mode": settings.retrieval_mode + (" + rerank" if settings.rerank else "")}
        try:
            with st.spinner("Retrieving and generating..."):
                pipeline = get_pipeline(settings)
                resp = pipeline.ask(question, file_hashes=selected_hashes, top_k=settings.top_k)
            entry.update(
                answer=resp.answer,
                abstained=resp.abstained,
                sources=list(resp.sources),
                chunks=[
                    {"text": c.text, "source": c.source, "page": c.page,
                     "score": float(c.score), "chunk_id": c.chunk_id, "rerank": c.rerank_score}
                    for c in resp.chunks
                ],
            )
        except APP_ERRORS as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            entry["error"] = f"Unexpected error: {type(exc).__name__}: {exc}"
        render_assistant(entry)
    st.session_state.messages.append(entry)

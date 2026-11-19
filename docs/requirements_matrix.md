# Requirements compliance matrix

Status as of 2026-11-20 (Version 1.3: hybrid search, reranking, header stripping; see the evaluation report §8e). Each status means:

- **Complete and verified**: implemented, covered by automated tests and/or observed in a real run (real embeddings, real Ollama `llama3.2:3b`).
- **Implemented, not verified**: code exists, but this project has not yet observed it working end to end (for example, a browser path that `AppTest` cannot drive).
- **Incomplete**: still to be done.
- **Optional enhancement**: discussed in class but not established as mandatory.
- **Needs professor clarification**: cannot be decided from the brief.

"Tests" refers to `tests/` (run with `.venv/bin/pytest -q`). "Eval" refers to the run files in `evaluation/results/` (see `docs/evaluation_report.md`).

## A. PDF ingestion and preprocessing

| # | Requirement | Status | Evidence |
|---|---|---|---|
| A1 | Upload one or more PDFs | Complete and verified | `app.py` (`st.file_uploader(..., accept_multiple_files=True)`); multi-file ingest in `tests/test_ingest.py::test_multiple_files`; the upload widget was driven in real Chrome (Playwright) for the demo recording: 4 files uploaded and indexed |
| A2 | Reject unsupported formats with a clear error | Complete and verified | `rag/pdf_loader.py::validate_pdf_bytes` (extension, `%PDF-` header, empty, size); tests in `tests/test_pdf_loader.py`; the audit live-tested a `.txt` renamed to `.pdf` |
| A3 | Extract text from text-based PDFs | Complete and verified | `PyPDFLoader` in `rag/pdf_loader.py::load_pdf_bytes`; all 7 dataset PDFs (245 pages) load with no empty pages |
| A4 | Preserve filename and page metadata | Complete and verified | `source`, 1-based `page`, `file_hash`, `chunk_id` metadata; `tests/test_pdf_loader.py`, `tests/test_chunking.py` |
| A5 | Empty pages, corrupted, encrypted, duplicate and no-text PDFs handled gracefully | Complete and verified | `tests/test_pdf_loader.py` (blank, corrupted, encrypted); `tests/test_ingest.py` (duplicate detected **before parsing** since V1.1, regression test `test_duplicate_detected_before_parsing`) |
| A6 | Normalise text without losing meaning | Complete and verified | `rag/pdf_loader.py::normalize_text`. V1.3: running headers/footers and page numbers stripped (`strip_repeated_lines`; audited on all 7 PDFs, tests in `tests/test_pdf_loader.py`). V1.1 expands ligatures (`ﬁ`→`fi`) and keeps real compounds when rejoining hyphenated line breaks (`English-to-German`, `self-attention`); regression cases in `tests/test_pdf_loader.py::test_normalize_text`. Known residual: some PDFs (Llama 2) come out of extraction with words glued together (`doubledthecontextlength`), which normalisation cannot repair |
| A7 | No OCR; explain when a PDF cannot be processed | Complete and verified | No-text PDF error says it is probably scanned/image-only and OCR is not supported (`rag/pdf_loader.py`); `tests/test_pdf_loader.py` |
| A8 | Dataset of about 50–200 pages, non-confidential, text-based | **Needs professor clarification** | Corpus A: 4 arXiv papers, 61 pages in total (each 11–19 pages). Corpus B (added in V1.1): GPT-3 (75 pp.) and Llama 2 (77 pp.), 152 pages in total. Both corpora fall within 50–200 pages if the range applies to the collection; corpus B also satisfies it per document. All six are public. Licences: Sentence-BERT is CC BY-SA 4.0; the other five are under the arXiv non-exclusive distribution licence (which grants redistribution rights only to arXiv). The input PDFs are to be shared (class session, 2026-11); `data/download_datasets.sh` re-downloads them. Corpus C (V1.2): NIST CSF 2.0, 32 pages, public domain, a different-domain check (report §8d) |

## B. Chunking

| # | Requirement | Status | Evidence |
|---|---|---|---|
| B1 | Overlapping chunks with a LangChain splitter | Complete and verified | `RecursiveCharacterTextSplitter` in `rag/chunking.py`; overlap test in `tests/test_chunking.py` |
| B2 | Configurable size and overlap | Complete and verified | `CHUNK_SIZE` / `CHUNK_OVERLAP` env vars (`rag/config.py`), UI inputs, `--chunk-size/--chunk-overlap` in `evaluation/run_eval.py` |
| B3 | Metadata preserved in chunks | Complete and verified | `tests/test_chunking.py` |
| B4 | Configuration recorded for evaluation | Complete and verified | Each results JSON stores a `config` snapshot (chunk size/overlap, top-k, models) |

## C. Embeddings

| # | Requirement | Status | Evidence |
|---|---|---|---|
| C1 | Pretrained embedding model for chunks and queries (the same model) | Complete and verified | `sentence-transformers/all-MiniLM-L6-v2` via `rag/embeddings.py::get_embeddings`; the same instance embeds both chunks and queries (`rag/vectorstore.py`) |
| C2 | Configurable model | Complete and verified | `EMBEDDING_MODEL` env var; `BAAI/bge-small-en-v1.5` was run through the same code in a retrieval-only experiment (see evaluation report §7) |
| C3 | Clear errors on initialisation or download failure | Complete and verified | `EmbeddingInitError` with an actionable message; tested |

## D. Vector database and retrieval

| # | Requirement | Status | Evidence |
|---|---|---|---|
| D1 | Store chunks, embeddings and metadata in a vector DB | Complete and verified | Persistent Chroma, cosine space (`rag/vectorstore.py`); `tests/test_vectorstore.py` |
| D2 | Semantic (not keyword-only) search | Complete and verified | Slow test with the real model retrieves paraphrased queries (`tests/test_retrieval.py`, marked `slow`) |
| D3 | Configurable top-k | Complete and verified | `TOP_K`, UI slider (1–10), `--top-k`; evaluated at k=4 and k=8 |
| D3b | Search type: hybrid (default), similarity or MMR | Complete and verified | `RETRIEVAL_MODE`, UI radio, `--retrieval-mode`; hybrid = BM25 + cosine with Reciprocal Rank Fusion (`VectorStoreManager.hybrid_search`); tests in `tests/test_vectorstore.py`; MMR evaluated in report §8b, hybrid in §8e |
| D3c | Cross-encoder reranking (V1.3, default on) | Complete and verified | `rag/reranker.py`, `RERANK*` settings, UI checkbox, `--rerank/--no-rerank`; `tests/test_retrieval.py`; wrong answers 7 → 2 of 22 (report §8e) |
| D4 | Retrieval restricted to the active corpus | Complete and verified | `file_hash` filter (`VectorStoreManager.similarity_search`); filter-isolation tests; the audit live-checked deselected documents (the system correctly fell back) |
| D5 | Reset / re-index without mixing documents | Complete and verified | `reset()`, `remove_file()`, UI "Clear corpus" (confirm-gated) and "Re-index with current settings"; V1.1 fix: a second store instance survives another instance's reset (`tests/test_vectorstore.py::test_second_instance_survives_reset_by_another`). Separate eval corpora use separate collections |
| D6 | Local store, no unnecessary infrastructure | Complete and verified | Chroma on local disk (`chroma_db/`, git-ignored) |

## E. LLM answer generation

| # | Requirement | Status | Evidence |
|---|---|---|---|
| E1 | Question plus retrieved context passed to the LLM | Complete and verified | `rag/generation.py::build_messages`; pipeline tests |
| E2 | Explicit system prompt restricting answers to the context | Complete and verified | `SYSTEM_PROMPT` (10 numbered rules, including prompt-injection hygiene and, since V1.1, rules about numbers in tables and false premises) |
| E2b | LLM parameters explicit and adjustable (V1.2) | Complete and verified | Temperature, top-p and context window (`num_ctx`) in `rag/config.py`, the UI sidebar and `run_eval.py` flags; `tests/test_generation.py::test_get_llm_passes_sampling_and_context_settings`; rationale in evaluation report §8c |
| E2c | Sampling top-k exposed (V1.3) | Complete and verified | `LLM_TOP_K` (default 40, Ollama's default), UI slider, `--llm-top-k`; test in `tests/test_generation.py` |
| E2d | Query decomposition for comparison questions (V1.3, optional) | Complete and verified; **off by default** | `decompose_question`, `QUERY_DECOMPOSITION`, UI checkbox, `--decompose`; tests in `tests/test_pipeline.py`; evaluated and not adopted as default (report §8e Step 4) |
| E3 | No silent use of general knowledge | Implemented, partially verified: a **known residual risk** | Enforced by the prompt. All 7 unanswerable questions (three corpora) were declined in every configuration tested. The A2 false premise produced a fabricated rationale at k=8 in V1 and V1.1; in V1.3 (top-k 4) it is corrected. A 3B model cannot guarantee this |
| E4 | Exact fallback when the context is insufficient | Complete and verified | `FALLBACK_ANSWER`; returned without calling the LLM when nothing is retrieved; observed for U1–U3 / LU1–LU2 |
| E5 | Partial answers say which part is unsupported | Implemented, partially verified | Prompt rule 4; question P1 was added to test this (see evaluation report); S1/S2 in V1 omitted parts without saying so |
| E6 | Source filename and page for substantive answers | Complete and verified | V1.1: the Sources line lists only the passages the answer cites (`rag/pipeline.py::cited_sources`), falling back to all retrieved pages if there are no `[n]` markers; abstentions show no sources; tests in `tests/test_pipeline.py` |
| E7 | Configurable provider; no hard-coded keys | Complete and verified | `LLM_PROVIDER=ollama|openai`; key read only from `OPENAI_API_KEY`; the audit found no secrets in the repo or git history. The OpenAI path is unit-tested only (no key available), so a live API call has not been verified |

## F. User interface (Streamlit)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| F1 | Multi-PDF upload | Complete and verified | See A1 (4 files uploaded and indexed in real Chrome for the demo) |
| F2 | Ingestion and indexing status | Complete and verified (rendering via `AppTest`) | `st.status` with progress and per-file results |
| F3 | Success / failure feedback | Complete and verified | Per-file indexed / duplicate / error messages |
| F4 | Chat interface with question and answer | Complete and verified | `st.chat_input` / `st.chat_message`; `tests/test_app_smoke.py`; the audit drove a real question through `AppTest` with live Ollama |
| F5 | Source references | Complete and verified | "Sources: file (p. N)" line, cited passages only (V1.1) |
| F6 | Inspect retrieved evidence | Complete and verified | "Retrieved evidence" expander: rank, source, page, score, chunk id, text |
| F7 | Clear / re-index corpus | Complete and verified (clear); re-index implemented, not verified in a browser | `tests/test_app_smoke.py` covers clear and remove |

## Code quality, testing, deliverables

| # | Requirement | Status | Evidence |
|---|---|---|---|
| Q1 | Modular code | Complete | `rag/{config,pdf_loader,chunking,embeddings,vectorstore,retrieval,reranker,ingest,generation,pipeline}.py`, `app.py`, `evaluation/run_eval.py`, `evaluation/retrieval_experiment.py` |
| Q2 | Automated tests for the critical components | Complete and verified | See the evaluation report §2 for the actual count and result |
| Q3 | `.gitignore`, dependency file, `.env.example`, no secrets | Complete | Dependencies use lower bounds only; a lock file is recommended (see "Remaining") |
| D-1 | Working app plus README | Complete and verified | `streamlit run app.py` starts (audit health check); README commands verified by the audit |
| D-2 | Architecture document and diagram | Complete | `docs/architecture.md` (Mermaid) |
| D-3 | Manual evaluation report, about 10–15 questions | Complete | `docs/evaluation_report.md`: 33 questions over three corpora (15 + 9 + 9), per-question judgements with rationale for every configuration (similarity, MMR, hybrid, hybrid + rerank; top-k 4/6/8; three LLMs) |
| D-4 | Presentation | Complete | `docs/presentation/RAG_PDF_Chatbot_Capstone.pptx` (16 slides with speaker notes; slide 8 has a real app screenshot; slide 12 covers MMR and LLM parameters; slide 13 the V1.3 improvements and LLM comparison) |
| D-5 | 5–10 minute demo video | Complete | `submission/RAG_PDF_Chatbot_Demo.mp4` (4:47, captioned screen recording of the final app; narration script in `docs/demo_voiceover.md`) |
| D-6 | Dataset included with the submission | Needs professor clarification (licensing) | See A8 |

## Optional / out of scope (not prerequisites)

| Item | Status | Note |
|---|---|---|
| Comparing 2–3 LLMs | Done (V1.3) | `llama3.2:3b` (default) vs `qwen2.5:3b` on all 33 questions and `llama3.1:8b` on corpus C, same retrieval (report §8e). An embedding comparison (MiniLM vs bge-small) was also run |
| MMR retrieval | Done (V1.2) | Implemented and evaluated (report §8b); optional, not the default |
| Reranker / hybrid BM25 search | Done (V1.3), default | Implemented after the evaluation identified retrieval misses as the main failure; measured gain in report §8e |
| Table-aware extraction | Tried, not adopted (V1.3) | `pdfplumber` returned figures as tables in the arXiv papers (report §8e Step 4) |
| Premise check before answering | Analysed, not built | Would add LLM calls to every question; false premises are already declined (report §8e Step 4) |
| Cloud deployment, authentication, multi-user, persistent chat memory | Out of scope | The class session confirmed a demo video is sufficient; local app |
| LLM-as-a-judge / automatic metrics | Optional enhancement | Manual evaluation was requested; no pass-rate threshold was specified |

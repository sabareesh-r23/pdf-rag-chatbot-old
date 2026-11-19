# PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions **only from the PDFs you
upload**, cites the file and page for every answer, shows the retrieved evidence, and replies with a
fixed fallback message when the documents do not contain the answer. Everything runs locally by
default (local embeddings + a local LLM via Ollama); any OpenAI-compatible API can be used instead.

Built as an academic capstone project. **Submission index:** [SUBMISSION.md](SUBMISSION.md) · **Demo video:** [submission/RAG_PDF_Chatbot_Demo.mp4](submission/RAG_PDF_Chatbot_Demo.mp4)

## Features

- Upload one or more PDFs in a Streamlit UI; per-file ingestion status (pages, chunks, skipped empty pages).
- Robust validation: non-PDF content (even if renamed to `.pdf`), empty, oversized, corrupted,
  encrypted and scanned/image-only PDFs are rejected with clear messages.
- Duplicate detection by SHA-256 file hash (re-uploads are not re-indexed).
- Persistent local vector store (Chroma) — the index survives app restarts.
- Active-corpus selection: only ticked documents are searched; remove single files or clear the corpus.
- Grounded answers with `[n]` citations, a **Sources** list (file + page of the passages the answer cites) and an expandable
  **Retrieved evidence** panel with similarity scores.
- Explicit fallback ("I could not find enough information in the uploaded documents to answer this
  question.") when the context does not contain the answer.
- **Hybrid retrieval with reranking (default):** BM25 keyword search and embedding search merged with
  Reciprocal Rank Fusion, then a cross-encoder reranks 30 candidates and keeps the best top-k.
  Similarity-only and MMR (Maximal Marginal Relevance) are selectable in the UI.
- Running page headers/footers and page numbers are stripped during PDF loading, so they do not
  distort retrieval.
- Adjustable top-k, chunk size / overlap and LLM parameters (temperature, top-p, context window) in the UI.
- Batch evaluation script producing JSON + Markdown records for manual grading.

## Architecture (summary)

```
Ingestion: upload -> validate -> PyPDF (per page, 1-based) -> normalize + strip running headers
           -> chunk (1000/200) -> embed (all-MiniLM-L6-v2) -> Chroma (cosine, metadata: source/page/file_hash)
Query:     question -> BM25 + embedding search over the active corpus, fused (RRF) -> 30 candidates
           -> cross-encoder rerank -> top-4 -> numbered context -> grounded prompt
           -> LLM (llama3.2:3b, temperature 0) -> answer + cited sources | fallback
```

See [docs/architecture.md](docs/architecture.md) for the Mermaid diagram, module responsibilities,
design rationale and limitations.

## Prerequisites

- **Python 3.12**
- **One LLM option:**
  - **Ollama (default, local, free):** install from <https://ollama.com/download> (macOS: the app, or
    `brew install ollama`), then:
    ```bash
    ollama serve            # not needed if the Ollama desktop app is running
    ollama pull llama3.2:3b # ~2 GB; fits an 8 GB Apple-Silicon Mac
    ```
  - **OpenAI-compatible API:** an API key (and optionally a base URL for a compatible server).
- ~3 GB free disk for dependencies (PyTorch is the largest) plus ~200 MB for the embedding and reranker
  models (downloaded automatically on first use).

## Setup

```bash
cd pdf-rag-chatbot

# Option A: uv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Option B: standard venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For the exact tested versions instead: pip install -r requirements-lock.txt

cp .env.example .env    # then edit if needed
```

## Run

```bash
streamlit run app.py
```

Open the URL printed by Streamlit (usually <http://localhost:8501>), upload PDFs in the sidebar, click
**Index uploaded PDFs**, and ask questions in the chat box. Sample papers are in `data/sample_pdfs/`.

## Configuration

All settings are environment variables (or entries in `.env`), defined in `rag/config.py`:

| Variable | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | `1000` | Chunk size in characters |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks (must be < `CHUNK_SIZE`) |
| `TOP_K` | `4` | Number of chunks retrieved per question |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` (BM25 + embeddings, RRF), `similarity` (embeddings only) or `mmr` (Maximal Marginal Relevance) |
| `RERANK` | `true` | Rerank `RERANK_CANDIDATES` retrieved chunks with a cross-encoder and keep the best `TOP_K` |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for reranking (~90 MB, downloaded on first use) |
| `RERANK_CANDIDATES` | `30` | Candidates passed to the reranker |
| `MMR_FETCH_K` | `20` | Candidate pool for MMR, and for each ranking in hybrid search |
| `MMR_LAMBDA` | `0.5` | MMR trade-off: 1 = relevance only, 0 = diversity only |
| `MIN_RELEVANCE` | `0.0` | Drop retrieved chunks with cosine similarity below this (0 = disabled) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local Sentence-Transformers model |
| `CHROMA_DIR` | `./chroma_db` | Directory of the persistent Chroma store |
| `COLLECTION_NAME` | `pdf_corpus` | Chroma collection used by the app |
| `MAX_FILE_MB` | `50` | Maximum upload size per PDF |
| `LLM_PROVIDER` | `ollama` | `ollama` or `openai` (any OpenAI-compatible API) |
| `LLM_MODEL` | `llama3.2:3b` | Model name for the chosen provider |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic) |
| `LLM_TOP_P` | `0.9` | Nucleus sampling; only has an effect when temperature > 0 |
| `LLM_NUM_CTX` | `4096` | Ollama context window in tokens (prompt + answer); ignored for `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` (read from env only, never stored) |
| `OPENAI_BASE_URL` | — | Optional: base URL of an OpenAI-compatible server |

Example — use an OpenAI-compatible API:

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
# OPENAI_BASE_URL=https://your-compatible-endpoint/v1
```

Top-k, search type (hybrid / similarity / MMR), reranking, temperature, top-p, context window, chunk
size and chunk overlap can also be changed at runtime in the sidebar **Settings**
(chunk changes apply to newly indexed files or via **Re-index with current settings**).

## Tests

```bash
pytest                 # full suite
pytest -m "not slow"   # skip tests that load the real embedding model (no network needed)
```

Tests use generated fixture PDFs and fake embeddings / chat models; no LLM is required.

## Evaluation

```bash
python -m evaluation.run_eval --questions evaluation/questions.json            # default config
python -m evaluation.run_eval --top-k 8                                        # vary top-k
python -m evaluation.run_eval --retrieval-mode mmr --mmr-lambda 0.5 --no-rerank # MMR retrieval
python -m evaluation.run_eval --retrieval-mode similarity --no-rerank            # V1.1 baseline retrieval
python -m evaluation.run_eval --chunk-size 500 --chunk-overlap 100 --reindex   # vary chunking
python -m evaluation.run_eval --llm-model llama3.1:8b                          # vary model
python -m evaluation.run_eval --ids F1,U1                                      # subset / smoke test
python -m evaluation.run_eval --questions evaluation/questions_long.json --pdf-dir data/long_pdfs   # long-document corpus
```

The script indexes the PDF folder into a separate collection per chunk configuration and corpus
(`eval_cs<size>_co<overlap>[_<folder>]`), runs every question and writes
`evaluation/results/results_<timestamp>.json` and `.md`. Each record holds the configuration, retrieved
chunks, answer, sources, abstention flag, latency and two automatic hints (expected page retrieved;
verbatim evidence snippets retrieved). Human judgement fields are left blank by the script.

Question sets (ground truth verified against the PDF text, with quotes and page numbers):
- `evaluation/questions.json`: 15 questions on `data/sample_pdfs/` (4 papers, 61 pages)
- `evaluation/questions_long.json`: 9 questions on `data/long_pdfs/` (GPT-3 and Llama 2 papers, 152 pages)
- `evaluation/questions_domain.json`: 9 questions on `data/domain_pdfs/` (NIST Cybersecurity Framework 2.0,
  32 pages): a different-domain check on a government policy document

**Results:** see [docs/evaluation_report.md](docs/evaluation_report.md). Judgements were made by
reading the retrieved chunks against verified expected answers. No results are claimed in this README.

## Project structure

```
pdf-rag-chatbot/
├── app.py                    # Streamlit UI
├── rag/
│   ├── config.py             # Settings from env, FALLBACK_ANSWER
│   ├── pdf_loader.py         # validation, per-page extraction, normalization
│   ├── chunking.py           # RecursiveCharacterTextSplitter, chunk ids
│   ├── embeddings.py         # local Sentence-Transformers embeddings (cached)
│   ├── vectorstore.py        # Chroma wrapper (cosine, dedupe, filtered search)
│   ├── ingest.py             # validate -> load -> chunk -> index orchestration
│   ├── retrieval.py          # hybrid / similarity / MMR retrieval over the active corpus, optional rerank
│   ├── reranker.py           # cross-encoder reranking (cached model)
│   ├── generation.py         # grounded prompt, LLM factory, answer generation
│   └── pipeline.py           # RAGPipeline.ask -> RAGResponse
├── evaluation/
│   ├── questions.json        # 15 verified questions (corpus A)
│   ├── questions_long.json   # 9 verified questions (corpus B, long documents)
│   ├── questions_domain.json # 9 verified questions (corpus C, NIST CSF 2.0)
│   ├── run_eval.py           # batch evaluation CLI
│   ├── retrieval_experiment.py  # retrieval-only comparison (report §8e)
│   └── results/              # final run files (+ superseded/ intermediate runs)
├── data/
│   ├── sample_pdfs/          # corpus A: 4 public arXiv papers, 61 pages
│   ├── long_pdfs/            # corpus B: GPT-3 + Llama 2 papers, 152 pages
│   ├── domain_pdfs/          # corpus C: NIST CSF 2.0, 32 pages (different domain)
│   └── download_datasets.sh  # re-download all seven PDFs
├── docs/
│   ├── architecture.md
│   ├── evaluation_report.md  # manual evaluation report (33 questions, 3 corpora)
│   ├── requirements_matrix.md
│   ├── evaluation_template.md  # blank template (superseded by evaluation_report.md)
│   ├── demo_walkthrough.md
│   ├── demo_voiceover.md     # timed narration script for the demo video
│   ├── submission_checklist.md
│   ├── audit_report.md       # independent audit of V1
│   └── presentation/         # RAG_PDF_Chatbot_Capstone.pptx + source/
├── submission/               # demo video (RAG_PDF_Chatbot_Demo.mp4)
├── SUBMISSION.md             # index of all deliverables
├── tests/                    # pytest suite (fixtures generated with reportlab)
├── requirements.txt
├── requirements-lock.txt     # exact tested versions
├── pytest.ini
└── .env.example
```

## Model and tool choices

| Component | Choice | Why |
|---|---|---|
| PDF extraction | PyPDF (`PyPDFLoader`) | Pure Python, per-page output for page citations; running headers stripped |
| Chunking | `RecursiveCharacterTextSplitter`, 1000 / 200 | Respects paragraph/sentence boundaries; one fact + context per chunk; overlap protects boundary facts |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Small (~90 MB), CPU-friendly; reads question and chunk together, which fixed most retrieval misses |
| Embeddings | `all-MiniLM-L6-v2` | Small (~90 MB), fast on CPU/Apple Silicon, free, strong for semantic search |
| Similarity | Cosine on normalized vectors | Matches how the model was trained; scores comparable across queries |
| Search type | Hybrid BM25 + embeddings, then cross-encoder rerank (default) | Cut wrong answers from 7 to 2 of 22 across three corpora (evaluation report §8e); MMR and similarity-only remain selectable |
| Vector store | Chroma (persistent, embedded) | No server, metadata filtering, LangChain integration |
| LLM | Ollama `llama3.2:3b`, temperature 0, top-p 0.9, context 4096 tokens | Runs locally on 8 GB RAM, private, free; deterministic output reduces hallucination. Compared with `qwen2.5:3b` (far more over-cautious with this prompt) and `llama3.1:8b` (reads better but ~4× slower and exceeds an 8 GB M1's GPU memory); see report §8e |
| LLM (optional) | Any OpenAI-compatible API | Larger models for comparison without code changes |
| UI | Streamlit | Fast to build an interactive upload + chat interface in Python |

## Data

Three document sets are used for the demo and the evaluation (`data/download_datasets.sh` re-downloads them):

- `data/sample_pdfs/` (corpus A, 61 pages): four public arXiv papers: Attention Is All You Need
  (1706.03762), BERT (1810.04805), Sentence-BERT (1908.10084) and Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks (2005.11401).
- `data/long_pdfs/` (corpus B, 152 pages): GPT-3 (2005.14165, 75 pp.) and Llama 2 (2307.09288, 77 pp.).
- `data/domain_pdfs/` (corpus C, 32 pages): NIST Cybersecurity Framework 2.0 (US government work, public
  domain), a different-domain check.

## Limitations

- No OCR: scanned/image-only PDFs are rejected; image-only pages are skipped.
- Tables, figures and equations are extracted as flat (sometimes garbled) text; multi-column layouts may
  interleave. A table extractor (`pdfplumber`) was tried and rejected: it returned figures as tables in the
  arXiv papers (evaluation report §8e).
- Chunks never span pages, so sentences across a page break are split.
- A 3B local model sometimes misreads correctly retrieved text (a wrong list item or step, a contradictory
  trailing fallback). This is now the main error source; `llama3.1:8b` reads better but did not fit an
  8 GB M1's GPU memory (evaluation report §8e).
- Some passages are still missed even with hybrid search and reranking (F3's p. 4 passage; the GPT-3 side
  of the cross-document question LM2).
- Some PDFs extract with words glued together (e.g. parts of the Llama 2 paper), which normalisation
  cannot repair.
- Single-user, local application (one shared index, no authentication).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar says Ollama is not running / connection refused | Start it: `ollama serve` (or open the Ollama app). Check `OLLAMA_BASE_URL`. |
| "Model ... is not pulled" | `ollama pull llama3.2:3b` (or the model in `LLM_MODEL`); verify with `ollama list`. |
| First start is slow / model download error | The embedding model and the reranker (~90 MB each) are downloaded from Hugging Face on first use and cached in `~/.cache/huggingface`. Ensure internet access for the first run and enough disk space. The app loads the reranker at start-up ("Loading reranker model..."). |
| Reranker cannot be loaded (offline) | Untick **Rerank with cross-encoder** in Settings, or set `RERANK=false`. |
| "probably a scanned or image-only PDF" error | The PDF has no text layer. OCR is out of scope; run it through an OCR tool (e.g. `ocrmypdf`) first. |
| "password-protected/encrypted" error | Upload an unencrypted copy. |
| Answers are slow or the Mac swaps | Close other apps; keep top-k small; untick reranking (saves ~2 s per answer); use a smaller model or an API provider. |
| Answers ignore a document | Make sure it is ticked in **Active corpus**. |
| `OPENAI_API_KEY is not set` | Set it in `.env` when `LLM_PROVIDER=openai`, or switch back to `ollama`. |
| Changed `CHUNK_SIZE` but nothing changed | Chunking applies at indexing time — use **Re-index with current settings** or clear the corpus and re-upload. |

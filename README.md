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
- Retrieval by similarity or **MMR** (Maximal Marginal Relevance), switchable in the UI.
- Adjustable top-k, chunk size / overlap and LLM parameters (temperature, top-p, context window) in the UI.
- Batch evaluation script producing JSON + Markdown records for manual grading.

## Architecture (summary)

```
Ingestion: upload -> validate -> PyPDF (per page, 1-based) -> normalize -> chunk (1000/200)
           -> embed (all-MiniLM-L6-v2) -> Chroma (cosine, metadata: source/page/file_hash)
Query:     question -> embed -> top-k similarity search filtered to active corpus
           -> numbered context -> grounded prompt -> LLM (temperature 0) -> answer + sources | fallback
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
- ~1 GB free disk for dependencies and the embedding model (downloaded automatically on first use).

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
| `RETRIEVAL_MODE` | `similarity` | `similarity` (plain top-k) or `mmr` (Maximal Marginal Relevance: relevant but non-redundant chunks) |
| `MMR_FETCH_K` | `20` | MMR candidate pool size |
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

Top-k, search type (similarity / MMR and its parameters), temperature, top-p, context window, chunk
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
python -m evaluation.run_eval --retrieval-mode mmr --mmr-lambda 0.5             # MMR retrieval
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
│   ├── retrieval.py          # top-k similarity or MMR retrieval over the active corpus
│   ├── generation.py         # grounded prompt, LLM factory, answer generation
│   └── pipeline.py           # RAGPipeline.ask -> RAGResponse
├── evaluation/
│   ├── questions.json        # 15 verified questions (corpus A)
│   ├── questions_long.json   # 9 verified questions (corpus B, long documents)
│   ├── run_eval.py           # batch evaluation CLI
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
│   ├── submission_checklist.md
│   ├── audit_report.md       # independent audit of V1
│   └── presentation/         # RAG_PDF_Chatbot_Capstone.pptx + source/
├── tests/                    # pytest suite (fixtures generated with reportlab)
├── requirements.txt
├── pytest.ini
└── .env.example
```

## Model and tool choices

| Component | Choice | Why |
|---|---|---|
| PDF extraction | PyPDF (`PyPDFLoader`) | Pure Python, per-page output for page citations |
| Chunking | `RecursiveCharacterTextSplitter`, 1000 / 200 | Respects paragraph/sentence boundaries; one fact + context per chunk; overlap protects boundary facts |
| Embeddings | `all-MiniLM-L6-v2` | Small (~90 MB), fast on CPU/Apple Silicon, free, strong for semantic search |
| Similarity | Cosine on normalized vectors | Matches how the model was trained; scores comparable across queries |
| Search type | Similarity (default); MMR optional | Similarity was better on the short-paper corpus; MMR was better on the long-document corpus (evaluation report §8b) |
| Vector store | Chroma (persistent, embedded) | No server, metadata filtering, LangChain integration |
| LLM | Ollama `llama3.2:3b`, temperature 0, top-p 0.9, context 4096 tokens | Runs locally on 8 GB RAM, private, free; deterministic output reduces hallucination |
| LLM (optional) | Any OpenAI-compatible API | Larger models for comparison without code changes |
| UI | Streamlit | Fast to build an interactive upload + chat interface in Python |

## Data

`data/sample_pdfs/` contains four public arXiv papers used as the evaluation dataset:
Attention Is All You Need (1706.03762), BERT (1810.04805), Sentence-BERT (1908.10084) and
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (2005.11401).

## Limitations

- No OCR: scanned/image-only PDFs are rejected; image-only pages are skipped.
- Tables, figures and equations are extracted as flat (sometimes garbled) text; multi-column layouts may interleave.
- Chunks never span pages, so sentences across a page break are split.
- A 3B local model can miss multi-part answers or occasionally ignore grounding instructions. In the
  evaluation it never corrected a false premise and once fabricated a rationale (see the evaluation report).
- Dense retrieval only (no keyword/BM25 hybrid or re-ranker): passages worded differently from the
  question can be missed (evaluation question F3).
- Some PDFs extract with words glued together (e.g. parts of the Llama 2 paper), which normalisation
  cannot repair.
- Single-user, local application (one shared index, no authentication).

## Troubleshooting

| Symptom | Fix |
|---|---|
| Sidebar says Ollama is not running / connection refused | Start it: `ollama serve` (or open the Ollama app). Check `OLLAMA_BASE_URL`. |
| "Model ... is not pulled" | `ollama pull llama3.2:3b` (or the model in `LLM_MODEL`); verify with `ollama list`. |
| First start is slow / embedding model download error | The embedding model (~90 MB) is downloaded from Hugging Face on first use and cached in `~/.cache/huggingface`. Ensure internet access for the first run and enough disk space. |
| "probably a scanned or image-only PDF" error | The PDF has no text layer. OCR is out of scope; run it through an OCR tool (e.g. `ocrmypdf`) first. |
| "password-protected/encrypted" error | Upload an unencrypted copy. |
| Answers are slow or the Mac swaps | Close other apps; keep top-k small; use a smaller model or an API provider. |
| Answers ignore a document | Make sure it is ticked in **Active corpus**. |
| `OPENAI_API_KEY is not set` | Set it in `.env` when `LLM_PROVIDER=openai`, or switch back to `ollama`. |
| Changed `CHUNK_SIZE` but nothing changed | Chunking applies at indexing time — use **Re-index with current settings** or clear the corpus and re-upload. |

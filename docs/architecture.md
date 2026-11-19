# Architecture

This document describes how the PDF RAG chatbot turns uploaded PDFs into grounded, cited answers,
which module owns each step, and why each design choice was made.

## 1. Data flow

```mermaid
flowchart TD
    subgraph Ingestion["Ingestion path (per uploaded file)"]
        U[User uploads PDF in Streamlit sidebar] --> V{validate_pdf_bytes<br/>.pdf extension, %PDF- header,<br/>non-empty, size <= MAX_FILE_MB}
        V -- invalid --> E1[IngestReport status=error<br/>user-facing message]
        V -- valid --> H[compute_file_hash sha256]
        H --> D{store.has_file?}
        D -- yes --> E2[IngestReport status=duplicate<br/>skip parsing and re-indexing]
        D -- no --> RD{pypdf opens it?<br/>not corrupted / not encrypted}
        RD -- no --> E1
        RD -- yes --> L[PyPDFLoader / pypdf<br/>one Document per page, 1-based page]
        L --> N[normalize_text<br/>expand ligatures, re-join hyphenated breaks<br/>keeping real compounds, collapse whitespace]
        N --> HS[strip_repeated_lines<br/>remove running headers/footers and page numbers<br/>kept once on the first page]
        HS --> S{any page with text?}
        S -- no --> E3[error: likely scanned / image-only PDF<br/>OCR out of scope]
        S -- yes --> C[split_documents<br/>RecursiveCharacterTextSplitter<br/>CHUNK_SIZE / CHUNK_OVERLAP, per page]
        C --> EM[get_embeddings<br/>all-MiniLM-L6-v2, normalized]
        EM --> DB[(Chroma persistent collection<br/>cosine space<br/>ids = chunk_id<br/>metadata: source, page, file_hash)]
    end

    subgraph Query["Query path (per question)"]
        Q[User question in chat input] --> QE[embed question<br/>same embedding model]
        QE --> SS[hybrid search default: BM25 + cosine rankings<br/>fused with Reciprocal Rank Fusion<br/>or similarity / MMR; filter: file_hash IN active corpus]
        DB -.-> SS
        SS --> RR[cross-encoder rerank<br/>30 candidates -> best top-k 4<br/>ms-marco-MiniLM-L-6-v2]
        RR --> G{MIN_RELEVANCE guard<br/>optional}
        G --> R{any chunks left?}
        R -- no --> FB[FALLBACK_ANSWER<br/>LLM not called]
        R -- yes --> CB[format_context<br/>numbered blocks: n, source, page]
        CB --> P[grounded prompt<br/>SYSTEM_PROMPT rules + context + question]
        P --> LLM[LLM, temperature 0, top-p 0.9, num_ctx 4096<br/>Ollama llama3.2:3b or OpenAI-compatible]
        LLM --> A[Answer with n citations]
        A --> OUT[UI: answer + sources file/page of cited passages<br/>+ expandable retrieved evidence with scores<br/>abstained flag if fallback]
        FB --> OUT
    end
```

## 2. Module responsibilities

| Module | Responsibility | Key public API |
|---|---|---|
| `rag/config.py` | All tunables from env / `.env`; validation; report snapshot; fallback sentence | `Settings`, `get_settings(**overrides)`, `FALLBACK_ANSWER` |
| `rag/pdf_loader.py` | Validate bytes (extension, header, size, encryption, corruption), extract text per page with 1-based page numbers, normalize text, detect empty / scanned pages | `validate_pdf_bytes`, `load_pdf_bytes`, `load_pdf_path`, `strip_repeated_lines`, `compute_file_hash`, `PDFValidationError`, `LoadResult` |
| `rag/chunking.py` | Split each page into overlapping chunks; add `chunk_index` and stable `chunk_id` | `split_documents(pages, chunk_size, chunk_overlap)` |
| `rag/embeddings.py` | Load and cache the local Sentence-Transformers model with normalized vectors; actionable errors on download/load failure | `get_embeddings(model_name)`, `EmbeddingInitError` |
| `rag/vectorstore.py` | Persistent Chroma collection (cosine); add/upsert chunks, list/remove files, reset corpus, filtered similarity, MMR and hybrid (BM25 + cosine, RRF) search, all returning cosine similarity | `VectorStoreManager` |
| `rag/ingest.py` | Orchestrate validate -> load -> dedupe -> chunk -> index; never raises for bad PDFs | `ingest_file(...) -> IngestReport` |
| `rag/retrieval.py` | Top-k hybrid, similarity or MMR search restricted to the active corpus, optional cross-encoder reranking of a larger candidate pool, typed results | `retrieve(...) -> list[RetrievedChunk]` |
| `rag/reranker.py` | Load and cache the cross-encoder; score (question, chunk) pairs | `get_reranker`, `rerank_scores`, `RerankerInitError` |
| `rag/generation.py` | Grounded system prompt, context formatting, LLM construction (Ollama / OpenAI-compatible), Ollama health check, answer generation with fallback | `SYSTEM_PROMPT`, `get_llm`, `check_ollama`, `format_context`, `generate_answer`, `LLMInitError` |
| `rag/pipeline.py` | End-to-end `ask`: retrieve -> relevance guard -> generate -> sources & abstention flag | `RAGPipeline.ask(...) -> RAGResponse` |
| `app.py` | Streamlit UI: upload & ingestion status, active-corpus selection, settings (top-k, search type and MMR parameters, reranking, temperature, top-p, context window, chunk size/overlap), LLM status, chat with sources and evidence, clear corpus | — |
| `evaluation/run_eval.py` | Batch evaluation over `evaluation/questions.json`, writes JSON + Markdown records for manual grading | CLI |

## 3. Design decisions and rationale

**PyPDFLoader / pypdf for extraction.** Pure-Python, no system dependencies, gives one document per
page so every chunk can be cited with a page number. It extracts the embedded text layer only (see
limitations).

**Text normalization.** PDF text often contains hyphenated line breaks (`embed-\nding`), typographic
ligatures from LaTeX (`ﬁ`, `ﬂ`), stray null bytes and irregular whitespace / blank lines. Ligatures are
expanded because the embedding tokenizer otherwise splits words such as `efﬁcient`. Hyphenated breaks
are re-joined unless the page itself uses the hyphenated form or the first part is capitalised/numeric,
so real compounds (`self-attention`, `English-to-German`) survive. Normalizing before chunking improves both embedding quality and how readable the quoted
evidence is in the UI.

**RecursiveCharacterTextSplitter, 1000 characters / 200 overlap (default).** The splitter prefers
paragraph, then line, then sentence boundaries, so chunks stay semantically coherent. About 1000
characters (~150–250 words) is large enough to hold a complete fact with its surrounding context
(e.g. a full hyper-parameter sentence or a table row with its caption), yet small enough that several
chunks fit in a small LLM's context window at top-k = 4. The 20% overlap reduces the chance that a fact
is cut in half at a chunk boundary. Chunks are produced **per page**, so each chunk maps to exactly one
page and citations stay precise; the trade-off is that a sentence spanning a page break is split.
Both values are configurable (env vars, UI, and `run_eval.py --chunk-size/--chunk-overlap`) so the
choice can be compared empirically.

**sentence-transformers/all-MiniLM-L6-v2 embeddings.** Runs locally on CPU/Apple Silicon (~90 MB,
384-dimensional), free, no API key, and is a strong general-purpose semantic-search model for its size.
Local embeddings mean document text never leaves the machine during indexing.

**Cosine similarity with normalized embeddings.** Sentence-Transformers models are trained with cosine
objectives; cosine ignores vector length and focuses on direction (meaning). The Chroma collection is
created with `hnsw:space = cosine`, and scores are reported as `1 - distance`, so higher = more similar
and values are comparable across queries — which makes the optional `MIN_RELEVANCE` threshold meaningful.

**Chroma persistent local store.** Embedded (no server to run), persists to `CHROMA_DIR` across app
restarts, supports metadata filtering, and integrates with LangChain.

**file_hash deduplication and active-corpus filter.** Every chunk carries the SHA-256 of its source
file. Re-uploading the same file (even under another name) is detected and skipped, so the index does
not fill with duplicates that would crowd out other evidence in top-k. Chunk ids are deterministic
(`<hash12>-p<page>-c<index>`), so re-adding is idempotent. At query time the search is filtered by
`file_hash IN (selected files)`, so the user controls exactly which documents can be cited and stale or
deselected documents can never leak into answers. "Clear corpus" deletes and recreates the collection.

**Top-k = 4.** Enough to cover facts that span two or three chunks while keeping the prompt short for a
3B-parameter model (long contexts dilute small models' attention and slow generation on 8 GB machines).
Adjustable in the UI and via `--top-k` for comparison.

**Hybrid search + cross-encoder reranking by default (V1.3).** Two retrieval failure causes were
verified in the evaluation: wording mismatch between question and passage (F3) and a key sentence that
is a small part of its chunk or is outscored by chunks sharing a running header (D1). Hybrid search adds
a BM25 keyword ranking to the embedding ranking and merges them with Reciprocal Rank Fusion, so exact
terms ("Wikipedia", "RS.MA") count. A cross-encoder then reads each of 30 candidates together with the
question and keeps the best 4, which is much sharper than comparing two separate embeddings. Measured on
26 questions, full evidence in the top 4 rose from 9 to 16; with the LLM, wrong answers fell from 7 to
2 of 22 (evaluation report §8e). BM25 is computed in memory over the active chunks (fine for a few
thousand chunks); the reranker adds about 1.5 s per question on an M1.

**Running headers stripped at load time (V1.3).** Lines repeated word for word at the top or bottom of
at least half of a document's pages, and bare page numbers, are removed except on the first page they
appear on. In the NIST document the running title matched every question that named the framework, so
header-carrying chunks crowded out the evidence (D1 ranked 55th).

**MMR as an option (V1.2).** MMR (Maximal Marginal Relevance) fetches the
20 most similar chunks and then picks k that are relevant but not near-duplicates of each other
(`VectorStoreManager.mmr_search`, lambda 0.5). It is implemented on the raw Chroma query so that every
returned chunk keeps its cosine similarity for the evidence panel. In the evaluation (report §8b) MMR
helped on the long-document corpus, where similarity search fills k with repeats of the same topic,
but hurt on the short papers: with 200-character chunk overlap, the continuation of a passage is in a
neighbouring chunk that MMR treats as redundant. It is kept as an option, not the default.

**Grounded prompt and explicit fallback.** The system prompt instructs the model to answer only from
numbered passages, cite them as `[n]`, state which part is missing for partial answers, treat passage
text as data (prompt-injection resistance) and otherwise reply with the exact `FALLBACK_ANSWER`
sentence. If retrieval returns nothing the LLM is not called at all. Two rules were added after
evaluation (V1.1): rule 9 tells the model to prefer numbers stated in sentences over rows of flattened
table text (root cause of a wrong BLEU value), and rule 10 tells it not to justify a premise that no
passage states (root cause of a fabricated answer to a false-premise question).

**Cited sources only.** The UI's Sources line lists the file/page of the passages the answer actually
cites (`[n]` markers), not every retrieved chunk; if an answer carries no valid markers it falls back to
all retrieved pages, and abstentions show no sources. All retrieved chunks remain inspectable in the
"Retrieved evidence" panel. The response's `abstained` flag is
derived from the fallback sentence, so abstention can be measured during evaluation.

**Temperature 0, top-p 0.9, context window 4096.** Deterministic, least "creative" decoding minimizes
hallucination and makes evaluation runs reproducible. With temperature 0 decoding is greedy, so top-p
(kept at Ollama's default) only matters if the temperature is raised. The context window is set
explicitly: Ollama cuts prompts to 4096 tokens by default on this machine, and the largest evaluated
prompt (top-k 8) is 2,545 tokens, so nothing is silently truncated. All three are adjustable in the UI.

**Ollama llama3.2:3b by default; any OpenAI-compatible API optionally.** The 3B model (~2 GB quantized)
runs comfortably on an 8 GB Apple-Silicon laptop, is free and private. `LLM_PROVIDER=openai` swaps in a
hosted or self-hosted OpenAI-compatible model (with `OPENAI_API_KEY` / `OPENAI_BASE_URL`) without code
changes — useful for comparing a small local model with a larger one.

**Configuration via environment.** Every tunable (chunking, top-k, models, thresholds, paths) is in
`rag/config.py` and recorded in each evaluation result, so reported results always state the exact
configuration that produced them.

## 4. Limitations

- **No OCR.** Scanned / image-only PDFs have no text layer; they are rejected with an explanatory
  message. Individual image-only pages are skipped and reported.
- **Tables, figures and equations.** Extraction flattens tables into text runs and loses figures;
  numeric answers that live only in tables may be retrieved poorly or misread. Mathematical notation is
  often garbled (e.g. sub/superscripts lost).
- **Multi-column layouts** (common in academic papers) can interleave text in extraction order.
- **Per-page chunking** splits sentences that cross a page break.
- **Small LLM quality.** A 3B model can miss facts spread over several chunks, paraphrase imprecisely,
  ignore citation instructions, or answer from prior knowledge despite the prompt. Use a larger model
  via `LLM_MODEL` / `LLM_PROVIDER=openai` when quality matters.
- **Pure dense retrieval.** No keyword/BM25 hybrid or re-ranker; exact identifiers, numbers and rare
  terms can be missed by the embedding model.
- **Single user, local.** One shared Chroma collection, no authentication, no concurrent-write
  handling; intended for local, single-user use.
- **English-centric** embedding model.

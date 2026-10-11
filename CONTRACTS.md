# Internal module contracts (for parallel implementation)

Project root: `~/dev/pdf-rag-chatbot`. Python 3.12 venv at `.venv` (use `.venv/bin/python`, `.venv/bin/pytest`).
Config lives in `rag/config.py` (already written — `Settings`, `get_settings(**overrides)`, `FALLBACK_ANSWER`). Do not change its public names.

All chunks are `langchain_core.documents.Document` objects. **Metadata keys (mandatory, exact names):**

| key | type | meaning |
|---|---|---|
| `source` | str | original PDF filename (basename only, e.g. `paper.pdf`) |
| `page` | int | **1-based** page number |
| `file_hash` | str | sha256 hex of file bytes (dedupe + corpus identity) |
| `chunk_id` | str | `f"{file_hash[:12]}-p{page}-c{index}"` (set by chunking) |
| `chunk_index` | int | index of chunk within its page |

## `rag/pdf_loader.py` (owner: ingestion agent)
```python
class PDFValidationError(Exception): ...          # user-facing message in str(e)

@dataclass
class LoadResult:
    filename: str
    file_hash: str
    pages: list[Document]          # one Document per page that has text (normalized)
    total_pages: int
    empty_pages: list[int]         # 1-based page numbers with no extractable text
    warnings: list[str]

def compute_file_hash(data: bytes) -> str
def validate_pdf_bytes(filename: str, data: bytes, max_file_mb: int = 50) -> None   # raises PDFValidationError
def normalize_text(text: str) -> str
def load_pdf_bytes(filename: str, data: bytes, max_file_mb: int = 50) -> LoadResult  # raises PDFValidationError
def load_pdf_path(path: str | Path, max_file_mb: int = 50) -> LoadResult
```
Rules: reject non-`.pdf` extension and bytes not starting with `%PDF-`; reject empty/oversized files; corrupted/encrypted -> PDFValidationError with clear text;
a PDF whose pages all lack text -> PDFValidationError explaining it is likely scanned/image-based and OCR is out of scope. Use LangChain `PyPDFLoader` (write bytes to a temp file) or pypdf directly; page metadata must be 1-based.

## `rag/chunking.py` (owner: ingestion agent)
```python
def split_documents(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]
```
Uses `RecursiveCharacterTextSplitter` (separators `["\n\n", "\n", ". ", " ", ""]`). Preserves all page metadata, adds `chunk_id`, `chunk_index`.

## `rag/embeddings.py` (owner: retrieval agent)
```python
class EmbeddingInitError(Exception): ...
def get_embeddings(model_name: str) -> Embeddings   # HuggingFaceEmbeddings, normalize_embeddings=True, cached (lru_cache); wraps failures in EmbeddingInitError with an actionable message
```

## `rag/vectorstore.py` (owner: retrieval agent)
```python
class VectorStoreManager:
    def __init__(self, settings: Settings, embeddings: Embeddings | None = None)   # Chroma persistent at settings.chroma_dir, collection settings.collection_name, cosine space
    def add_chunks(self, chunks: list[Document]) -> int          # ids = chunk_id; returns number added
    def has_file(self, file_hash: str) -> bool
    def list_files(self) -> list[dict]                           # [{"source","file_hash","chunks"}]
    def count(self) -> int
    def reset(self) -> None                                       # deletes and recreates collection (clean corpus)
    def remove_file(self, file_hash: str) -> int
    def similarity_search(self, query: str, k: int, file_hashes: list[str] | None = None) -> list[tuple[Document, float]]
        # score = cosine similarity in [0,1]-ish (higher is better); filter on file_hash $in when given
```

## `rag/retrieval.py` (owner: retrieval agent)
```python
@dataclass
class RetrievedChunk:
    text: str; source: str; page: int; score: float; chunk_id: str
def retrieve(store: VectorStoreManager, question: str, k: int, file_hashes: list[str] | None = None) -> list[RetrievedChunk]
```
Raises `ValueError` on empty/whitespace question.

## `rag/ingest.py` (owner: retrieval agent)
```python
@dataclass
class IngestReport:
    filename: str; status: str   # "indexed" | "duplicate" | "error"
    message: str; pages: int = 0; empty_pages: list[int] = field(default_factory=list); chunks: int = 0; file_hash: str = ""
def ingest_file(store: VectorStoreManager, filename: str, data: bytes, settings: Settings) -> IngestReport   # never raises for bad PDFs; returns status="error"
```

## `rag/generation.py` (owner: generation agent)
```python
SYSTEM_PROMPT: str
class LLMInitError(Exception): ...
def get_llm(settings: Settings) -> BaseChatModel        # ChatOllama or ChatOpenAI; clear error if unavailable
def format_context(chunks: list[RetrievedChunk]) -> str  # numbered blocks "[1] (source: x.pdf, page 3)\n<text>"
def generate_answer(llm, question: str, chunks: list[RetrievedChunk]) -> str  # returns FALLBACK_ANSWER without calling LLM if chunks empty
```

## `rag/pipeline.py` (owner: generation agent)
```python
@dataclass
class RAGResponse:
    question: str; answer: str; chunks: list[RetrievedChunk]; sources: list[dict]   # sources: unique [{"source","page"}] in rank order
    abstained: bool                                   # True when answer is (or starts with) the fallback
    config: dict
class RAGPipeline:
    def __init__(self, settings: Settings, store: VectorStoreManager, llm=None)   # llm lazy via get_llm if None
    def ask(self, question: str, file_hashes: list[str] | None = None, top_k: int | None = None) -> RAGResponse
```
Optional relevance guard: `MIN_RELEVANCE` env (float, default 0.0 = disabled) — drop chunks below threshold.

## `app.py` (owner: UI agent) — Streamlit, uses only the public APIs above.
## `evaluation/` (owner: eval/docs agent) — `run_eval.py` reads `evaluation/questions.json`, runs RAGPipeline, writes JSON+Markdown with retrieved chunks.

Tests: `tests/test_<module>.py`, pytest, no network/LLM required (use fake embeddings e.g. `langchain_core.embeddings.FakeEmbeddings` or a deterministic bag-of-words embedding in `tests/conftest.py`, and a fake chat model). Fixture PDFs generated with reportlab in `tests/conftest.py` (owner: ingestion agent creates `conftest.py`; others may append fixtures but must not remove existing ones).

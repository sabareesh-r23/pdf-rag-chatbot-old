"""Persistent Chroma vector store wrapper for the PDF corpus."""
from __future__ import annotations

import functools
import math
import re
from collections import Counter
from collections import OrderedDict

import numpy as np
from chromadb.errors import NotFoundError

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from rag.config import Settings
from rag.embeddings import get_embeddings

# Chroma rejects very large single batches; stay well below its limit.
_ADD_BATCH_SIZE = 1000
_GET_BATCH_SIZE = 5000
# Reciprocal Rank Fusion constant (Cormack et al., 2009); 60 is the standard value.
_RRF_K = 60
_TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by does did do for from has have how in is it its of on or that the this to "
    "was were what when where which who why with according paper document documents".split()
)


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens for BM25 (keeps identifiers such as rs.ma, gpt-3, 2048).

    Single letters are dropped: the "s" of a possessive ("company's") would otherwise
    match unrelated chunks ("automobile's") and win the keyword ranking on its own.
    """
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]


def _bm25_scores(query: list[str], docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Okapi BM25 score of every document for the query tokens."""
    n = len(docs)
    if n == 0:
        return []
    avgdl = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d))
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in set(query)}
    scores = []
    for d in docs:
        tf = Counter(d)
        dl = len(d)
        score = 0.0
        for t in query:
            f = tf.get(t, 0)
            if f:
                score += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)
    return scores


def _reopen_on_missing(method):
    """Retry once with a fresh handle if another instance reset (deleted) the collection."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except NotFoundError:
            self._store = self._open()
            return method(self, *args, **kwargs)

    return wrapper


class VectorStoreManager:
    """Thin layer over langchain_chroma.Chroma with corpus-management helpers."""

    def __init__(self, settings: Settings, embeddings: Embeddings | None = None):
        self.settings = settings
        self.embeddings = embeddings if embeddings is not None else get_embeddings(settings.embedding_model)
        self._store = self._open()

    def _open(self) -> Chroma:
        return Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.chroma_dir,
            # Cosine distance so that similarity = 1 - distance.
            collection_metadata={"hnsw:space": "cosine"},
        )

    @property
    def _collection(self):
        return self._store._collection

    # ------------------------------------------------------------------ writes
    @_reopen_on_missing
    def add_chunks(self, chunks: list[Document]) -> int:
        """Add chunks using chunk_id as the id, so re-adding a file is idempotent (upsert)."""
        if not chunks:
            return 0
        # Deduplicate ids within the call (Chroma errors on duplicate ids in one batch).
        unique: OrderedDict[str, Document] = OrderedDict()
        for doc in chunks:
            chunk_id = doc.metadata.get("chunk_id")
            if not chunk_id:
                raise ValueError("Every chunk must have a 'chunk_id' in its metadata")
            unique[chunk_id] = doc
        ids = list(unique.keys())
        docs = list(unique.values())
        for start in range(0, len(docs), _ADD_BATCH_SIZE):
            self._store.add_documents(
                docs[start : start + _ADD_BATCH_SIZE], ids=ids[start : start + _ADD_BATCH_SIZE]
            )
        return len(docs)

    def reset(self) -> None:
        """Delete the whole collection and recreate it empty (no stale chunks survive)."""
        self._store.delete_collection()
        self._store = self._open()

    @_reopen_on_missing
    def remove_file(self, file_hash: str) -> int:
        """Delete all chunks of one file; returns the number of chunks removed."""
        ids = self._collection.get(where={"file_hash": file_hash}, include=[])["ids"]
        if ids:
            for start in range(0, len(ids), _ADD_BATCH_SIZE):
                self._collection.delete(ids=ids[start : start + _ADD_BATCH_SIZE])
        return len(ids)

    # ------------------------------------------------------------------- reads
    @_reopen_on_missing
    def count(self) -> int:
        return self._collection.count()

    @_reopen_on_missing
    def has_file(self, file_hash: str) -> bool:
        result = self._collection.get(where={"file_hash": file_hash}, limit=1, include=[])
        return bool(result["ids"])

    @_reopen_on_missing
    def list_files(self) -> list[dict]:
        """Aggregate chunk metadata into one entry per indexed file."""
        files: OrderedDict[str, dict] = OrderedDict()
        total = self.count()
        for offset in range(0, total, _GET_BATCH_SIZE):
            batch = self._collection.get(include=["metadatas"], limit=_GET_BATCH_SIZE, offset=offset)
            for meta in batch["metadatas"] or []:
                meta = meta or {}
                file_hash = meta.get("file_hash", "")
                entry = files.setdefault(
                    file_hash, {"source": meta.get("source", "unknown"), "file_hash": file_hash, "chunks": 0}
                )
                entry["chunks"] += 1
        return sorted(files.values(), key=lambda f: f["source"].lower())

    @_reopen_on_missing
    def similarity_search(
        self, query: str, k: int, file_hashes: list[str] | None = None
    ) -> list[tuple[Document, float]]:
        """Return (document, cosine similarity) pairs, best first.

        file_hashes=None searches the whole collection; an empty list means
        "no documents selected" and returns nothing.
        """
        if file_hashes is not None and len(file_hashes) == 0:
            return []
        if self.count() == 0:
            return []
        where = self._where(file_hashes)
        results = self._store.similarity_search_with_score(query, k=k, filter=where)
        # Chroma returns cosine *distance*; convert to similarity (higher = better).
        return [(doc, 1.0 - float(distance)) for doc, distance in results]

    @staticmethod
    def _where(file_hashes: list[str] | None) -> dict | None:
        if not file_hashes:
            return None
        if len(file_hashes) == 1:
            return {"file_hash": file_hashes[0]}
        return {"file_hash": {"$in": list(file_hashes)}}

    @_reopen_on_missing
    def mmr_search(
        self,
        query: str,
        k: int,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        file_hashes: list[str] | None = None,
    ) -> list[tuple[Document, float]]:
        """Maximal Marginal Relevance search, returning (document, cosine similarity) pairs.

        Fetches the `fetch_k` most similar chunks, then greedily picks `k` of them that
        balance similarity to the query (weight `lambda_mult`) against similarity to the
        chunks already picked (weight 1 - `lambda_mult`). This avoids filling the context
        with near-duplicate chunks (e.g. overlapping neighbours from one page).
        Pairs are returned in MMR selection order; the score is the query similarity,
        so it is comparable with `similarity_search`.
        """
        if file_hashes is not None and len(file_hashes) == 0:
            return []
        if self.count() == 0:
            return []
        query_embedding = self.embeddings.embed_query(query)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=max(fetch_k, k),
            where=self._where(file_hashes),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = result["ids"][0]
        if not ids:
            return []
        embeddings = result["embeddings"][0]
        selected = maximal_marginal_relevance(
            np.asarray(query_embedding, dtype=np.float32),
            [list(e) for e in embeddings],
            lambda_mult=lambda_mult,
            k=min(k, len(ids)),
        )
        out: list[tuple[Document, float]] = []
        for i in selected:
            doc = Document(
                page_content=result["documents"][0][i] or "",
                metadata=dict(result["metadatas"][0][i] or {}),
                id=ids[i],
            )
            out.append((doc, 1.0 - float(result["distances"][0][i])))
        return out

    @_reopen_on_missing
    def hybrid_search(
        self, query: str, k: int, file_hashes: list[str] | None = None, fetch_k: int = 20
    ) -> list[tuple[Document, float]]:
        """Hybrid keyword + semantic search, returning (document, cosine similarity) pairs.

        Ranks the active chunks twice, by cosine similarity of the embeddings (meaning)
        and by BM25 (exact words), and merges the two rankings with Reciprocal Rank
        Fusion over the top ``fetch_k`` of each. Exact terms such as "Wikipedia",
        "RS.MA" or "Core Functions" can therefore pull in a chunk that the embedding
        alone ranks low. The returned score is the cosine similarity, so it is
        comparable with `similarity_search`.
        """
        if file_hashes is not None and len(file_hashes) == 0:
            return []
        if self.count() == 0:
            return []
        where = self._where(file_hashes)
        ids, texts, metas, embs = [], [], [], []
        total = self.count()
        for offset in range(0, total, _GET_BATCH_SIZE):
            batch = self._collection.get(
                where=where, include=["documents", "metadatas", "embeddings"],
                limit=_GET_BATCH_SIZE, offset=offset,
            )
            ids += batch["ids"]
            texts += [t or "" for t in batch["documents"]]
            metas += [m or {} for m in batch["metadatas"]]
            embs += list(batch["embeddings"])
        if not ids:
            return []
        q = np.asarray(self.embeddings.embed_query(query), dtype=np.float32)
        m = np.asarray(embs, dtype=np.float32)
        cos = (m @ q) / (np.linalg.norm(m, axis=1) * (np.linalg.norm(q) or 1.0) + 1e-12)
        bm25 = _bm25_scores(_tokens(query), [_tokens(t) for t in texts])
        pool = max(fetch_k, k)
        dense_rank = [int(i) for i in np.argsort(-cos)[:pool]]
        # Only chunks that share at least one query word take part in the keyword ranking.
        keyword_rank = [i for i in sorted(range(len(ids)), key=lambda i: -bm25[i]) if bm25[i] > 0][:pool]
        fused: dict[int, float] = {}
        for ranking in (dense_rank, keyword_rank):
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (_RRF_K + rank)
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [
            (Document(page_content=texts[i], metadata=dict(metas[i]), id=ids[i]), float(cos[i]))
            for i in best
        ]

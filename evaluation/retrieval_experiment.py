"""Retrieval-only experiment (no LLM): does the verified evidence reach the top-k?

For every question with ``evidence_snippets`` (answerable and false-premise questions),
it checks whether *all* snippets appear in the text of the top 4 / top 8 chunks, for
four retrieval options: similarity, hybrid (BM25 + embeddings), and each of them
followed by cross-encoder reranking of 30 candidates. Uses the evaluation collections
built by ``run_eval.py`` (run that first). Reported in docs/evaluation_report.md §8e.

The snippet check is strict (it has false negatives when a chunk words a number
slightly differently), but it is applied identically to every option.

    python -m evaluation.retrieval_experiment
"""
from __future__ import annotations

import json
import re

from evaluation.run_eval import eval_collection_name
from rag.config import PROJECT_ROOT, get_settings
from rag.reranker import rerank_scores
from rag.vectorstore import VectorStoreManager

CANDIDATES = 30
CORPORA = [
    ("A", "evaluation/questions.json", "data/sample_pdfs"),
    ("B", "evaluation/questions_long.json", "data/long_pdfs"),
    ("C", "evaluation/questions_domain.json", "data/domain_pdfs"),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def main() -> None:
    base = get_settings()
    totals: dict[tuple[str, int], list[bool]] = {}
    for corpus, qfile, pdf_dir in CORPORA:
        store = VectorStoreManager(get_settings(collection_name=eval_collection_name(base, PROJECT_ROOT / pdf_dir)))
        questions = json.loads((PROJECT_ROOT / qfile).read_text())
        for q in questions:
            snippets = q.get("evidence_snippets") or []
            if not snippets or q["category"] == "unanswerable":
                continue
            pools = {
                "similarity": store.similarity_search(q["question"], k=CANDIDATES),
                "hybrid": store.hybrid_search(q["question"], k=CANDIDATES, fetch_k=CANDIDATES),
            }
            row = []
            for name, pool in pools.items():
                scores = rerank_scores(base.rerank_model, q["question"], [d.page_content for d, _ in pool])
                reranked = [pool[i] for i in sorted(range(len(pool)), key=lambda i: -scores[i])]
                for label, docs in ((name, pool), (name + " + rerank", reranked)):
                    for k in (4, 8):
                        text = _norm(" ".join(d.page_content for d, _ in docs[:k]))
                        hit = all(_norm(s) in text for s in snippets)
                        totals.setdefault((label, k), []).append(hit)
                        if k == 4:
                            row.append(f"{label}={'Y' if hit else '.'}")
            print(corpus, q["id"], " ".join(row))
    print()
    for (label, k), hits in sorted(totals.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"{label:20s} top-{k}: full evidence for {sum(hits)}/{len(hits)} questions")


if __name__ == "__main__":
    main()

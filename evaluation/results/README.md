# Evaluation run files

Each run has a `.json` (full record: retrieved chunks with page, similarity and rerank scores, answer,
sources, latency, configuration) and a `.md` (the same, readable). They are produced by
`python -m evaluation.run_eval`; the manual judgements are in `docs/evaluation_report.md`.

**Start here: the final code with its default configuration (hybrid search + reranker, top-k 4,
`llama3.2:3b`, temperature 0):**

| Corpus | Run |
|---|---|
| A: 4 arXiv papers, 15 questions | `results_20261120_010040` |
| B: 2 long papers, 9 questions | `results_20261120_010255` |
| C: NIST CSF 2.0, 9 questions | `results_20261120_010426` |

Every other run (baseline, MMR, ablations, top-k, other LLMs, query decomposition) is listed with its
purpose in `docs/evaluation_report.md` §3. `superseded/` holds runs from partly fixed intermediate code,
with the reason each was replaced.

# Manual Evaluation Report — PDF RAG Chatbot

> **NOT YET COMPLETED — fill in from actual run outputs.**
> Every cell marked `TBD` must be filled from a real `evaluation/results/results_<timestamp>.md/.json`
> file produced by `python -m evaluation.run_eval`. Do not estimate or invent results.

## 1. Methodology

**Goal.** Measure whether the chatbot (a) retrieves the passage that contains the answer, (b) answers
correctly and only from that evidence, and (c) abstains when the documents do not contain the answer.

**Dataset.** Four public arXiv papers in `data/sample_pdfs/`:

| File | Paper | Pages |
|---|---|---|
| `arxiv_1706.03762.pdf` | Vaswani et al., *Attention Is All You Need* (2017) | 15 |
| `arxiv_1810.04805.pdf` | Devlin et al., *BERT* (2018) | 16 |
| `arxiv_1908.10084.pdf` | Reimers & Gurevych, *Sentence-BERT* (2019) | 11 |
| `arxiv_2005.11401.pdf` | Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) | 19 |

**Question set.** `evaluation/questions.json` — 14 questions in five categories:

| Category | IDs | What it tests |
|---|---|---|
| factual | F1–F4 | single fact located in one passage |
| multi_chunk | M1–M3 | answer needs facts from several chunks / pages |
| synthesis | S1–S2 | explanation combining several passages (S2 spans two papers) |
| unanswerable | U1–U3 | plausible question whose answer is absent from the corpus; correct behaviour is abstention |
| adversarial | A1–A2 | false premise; correct behaviour is to correct the premise from the documents (or abstain), not to invent |

Ground truth was verified against text extracted with `pypdf` from the actual PDFs; every answerable
question's `notes` field contains a verbatim quote with its 1-based page number. Absence for
unanswerable questions was verified by regex search over all 61 extracted pages.

**Procedure.**
1. Start Ollama (or configure an OpenAI-compatible provider) and run
   `python -m evaluation.run_eval --questions evaluation/questions.json --reindex`.
2. The script indexes all four PDFs into a dedicated collection (`eval_cs<size>_co<overlap>`), asks each
   question through `RAGPipeline.ask`, and records the retrieved chunks (rank, source, page, score, text),
   the answer, the cited sources, the abstention flag, latency, and the automatic hint
   `expected_page_retrieved`.
3. A human evaluator reads each answer next to its retrieved chunks and fills in the judgement columns
   below using the definitions in Section 4.

**Metrics** (computed by hand from the table once completed):
- *Retrieval hit rate* = answerable questions where retrieved context contains the evidence / answerable questions.
- *Answer accuracy* = Correct = Y / answerable questions.
- *Groundedness* = Supported = Y / answered (non-abstained) questions.
- *Abstention accuracy* = Correct abstention = Y / unanswerable questions; also count false abstentions on answerable questions.
- *Mean latency* (from the results file).

## 2. Configuration under test

| Setting | Value |
|---|---|
| Results file | TBD (`evaluation/results/results_<timestamp>.json`) |
| Date of run | TBD |
| LLM provider / model | TBD (default `ollama` / `llama3.2:3b`) |
| LLM temperature | TBD (default 0.0) |
| Embedding model | TBD (default `sentence-transformers/all-MiniLM-L6-v2`) |
| Chunk size / overlap | TBD (default 1000 / 200) |
| Top-k | TBD (default 4) |
| MIN_RELEVANCE | TBD (default 0.0 = disabled) |
| Total chunks indexed | TBD |
| Hardware | TBD (e.g. Apple M1, 8 GB RAM) |

## 3. Per-question results

Legend: Y / N / Partial; "Correct abstention?" is N/A for answerable questions (F, M, S). For A1–A2
record whether the false premise was corrected. "Retrieved context contains evidence?" — check the
retrieved chunks yourself; the `expected_page_retrieved` hint in the results file is a starting point
only (the right page can be retrieved without the right chunk).

| Test ID | Question | Category | Expected answer (verified) | Expected page(s) | Retrieved context contains evidence? (Y/N) | Actual response | Correct? | Relevant? | Supported by context? | Correct abstention? (N/A where answerable) | Failure class (Retrieval / Context construction / Generation / Abstention / None) | Observations & improvement |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F1 | Transformer encoder: number of layers and d_model? | factual | N = 6 identical layers; d_model = 512 | 1706.03762 p. 3 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| F2 | Total parameters of BERT-Base and BERT-Large? | factual | 110M (L=12,H=768,A=12) and 340M (L=24,H=1024,A=16) | 1810.04805 p. 3, 8 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| F3 | RAG non-parametric memory source and index size? | factual | Dec 2018 Wikipedia dump split into 100-word chunks, 21M documents (FAISS MIPS index) | 2005.11401 p. 4 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| F4 | Most-similar pair in 10,000 sentences: BERT vs SBERT time? | factual | ~65 hours with BERT vs ~5 seconds with SBERT | 1908.10084 p. 1, 2 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| M1 | Big Transformer BLEU (EN-DE, EN-FR) and training time/hardware? | multi_chunk | 28.4 EN-DE, 41.8 EN-FR; 3.5 days on 8 P100 GPUs (300k steps) | 1706.03762 p. 1, 7, 8 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| M2 | BERT's two pre-training tasks incl. masking and sentence pairs? | multi_chunk | MLM: 15% positions; 80% [MASK] / 10% random / 10% unchanged. NSP: 50% IsNext / 50% NotNext | 1810.04805 p. 4, 12 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| M3 | RAG-Sequence vs RAG-Token difference? | multi_chunk | Sequence: same document for whole output; Token: different document per token, marginalized per token | 2005.11401 p. 3, 4 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| S1 | Why raw BERT embeddings are poor and how SBERT fixes it? | synthesis | Cross-encoder infeasible (~65 h); avg/CLS embeddings worse than GloVe (54.81 / 29.19); SBERT = pooling (default MEAN) + siamese/triplet fine-tuning for cosine-comparable embeddings | 1908.10084 p. 1, 3, 4 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| S2 | How RAG builds on BERT; what generator it uses? | synthesis | DPR bi-encoder with BERT-Base document and query encoders (110M each; doc encoder/index fixed); generator BART-large (~400M) | 2005.11401 p. 3, 4, 18; 1810.04805 p. 3 | TBD | TBD | TBD | TBD | TBD | N/A | TBD | TBD |
| U1 | How many parameters does GPT-4 have? | unanswerable | Not in documents — abstain | — | N/A | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| U2 | RAG accuracy on MMLU? | unanswerable | Not in documents — abstain | — | N/A | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| U3 | Tonnes of CO2 emitted training the big Transformer? | unanswerable | Not in documents — abstain (training time is given, emissions are not) | — | N/A | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A1 | Why did the Transformer use LSTM layers in its encoder? | adversarial | False premise — no recurrence; attention only | 1706.03762 p. 1, 2 | TBD | TBD | TBD | TBD | TBD | TBD (premise corrected?) | TBD | TBD |
| A2 | Why does SBERT use [CLS] as default pooling? | adversarial | False premise — default pooling is MEAN | 1908.10084 p. 3 | TBD | TBD | TBD | TBD | TBD | TBD (premise corrected?) | TBD | TBD |

### Summary (fill in after the table)

| Metric | Value |
|---|---|
| Retrieval hit rate (answerable, 11) | TBD |
| Answer accuracy (answerable, 11) | TBD |
| Groundedness (answered questions) | TBD |
| Abstention accuracy (unanswerable, 3) | TBD |
| False abstentions on answerable questions | TBD |
| False premises corrected (adversarial, 2) | TBD |
| Mean latency (s) | TBD |
| Failure class counts (R / C / G / A) | TBD |

## 4. Failure class definitions

| Class | Definition | Typical evidence | Typical fixes |
|---|---|---|---|
| **Retrieval** | The chunk(s) containing the answer were **not** among the top-k retrieved chunks. | Expected page / passage missing from the retrieved list; low scores. | Increase top-k; smaller chunks; better embedding model; hybrid BM25 + dense search; re-ranker; query rewriting. |
| **Context construction** | The right evidence was retrieved but the context given to the LLM was inadequate: the fact was split across chunk boundaries, truncated, garbled by extraction (tables, equations, two-column order), crowded out by near-duplicates, or dropped by the relevance threshold. | Evidence present only partially, or text is unreadable in the retrieved chunk. | Adjust chunk size/overlap; better PDF parsing; deduplicate; tune `MIN_RELEVANCE`; order/label passages better. |
| **Generation** | The evidence was present and readable in the context, but the LLM answered wrongly: misread or ignored it, combined facts incorrectly, hallucinated details, answered from prior knowledge, or cited wrong passages. | Answer contradicts or goes beyond the retrieved text. | Stronger/larger model; tighter prompt; fewer, more relevant chunks; temperature 0. |
| **Abstention** | Wrong abstention behaviour: answered an unanswerable question (or accepted a false premise) instead of falling back, **or** returned the fallback although the evidence was in the context (false abstention). | `abstained` flag disagrees with the expected behaviour. | Prompt wording for fallback; relevance threshold; larger model; post-hoc groundedness check. |
| **None** | Correct, relevant, supported answer (or correct abstention). | — | — |

If several apply, record the **earliest** failing stage in the pipeline (Retrieval -> Context
construction -> Generation/Abstention) and mention the others in *Observations*.

## 5. How to complete this report

1. Run the evaluation (see Section 1). Note the output file names in Section 2 and copy the
   configuration values from the results file's *Configuration* table.
2. For each question, open its section in `results_<timestamp>.md`:
   - Read the retrieved chunks. Mark **Retrieved context contains evidence?** = Y only if the actual
     supporting text (see the quote in *Ground-truth notes*) is inside a retrieved chunk.
   - Paste (or faithfully abbreviate) the **Actual response**.
   - **Correct?** — does the answer match the verified expected answer? (Y / Partial / N)
   - **Relevant?** — does it address the question asked? (Y / N)
   - **Supported by context?** — is every claim backed by the retrieved chunks, with sensible
     citations? (Y / N). Correct-but-unsupported (from model memory) counts as N.
   - **Correct abstention?** — for U1–U3: Y if the fallback was returned; for A1–A2: Y if the premise
     was corrected or the system abstained; N/A for answerable questions.
   - **Failure class** — per Section 4.
   - **Observations & improvement** — what happened and a concrete fix to try.
3. Optionally copy the judgements back into the `human_judgement` fields of the JSON file to keep a
   machine-readable record.
4. Compute the summary metrics. Report counts with denominators (e.g. "9/11"), not just percentages.
5. Write a short discussion: dominant failure class, examples, and which improvement you would try first.
6. Remove the "NOT YET COMPLETED" banner only when every `TBD` is replaced with observed data.

## 6. Optional: configuration comparison

Run the evaluation once per configuration (each chunk configuration uses its own collection, so runs
do not interfere), e.g.:

```bash
python -m evaluation.run_eval --top-k 2
python -m evaluation.run_eval --top-k 8
python -m evaluation.run_eval --chunk-size 500 --chunk-overlap 100 --reindex
python -m evaluation.run_eval --chunk-size 1500 --chunk-overlap 300 --reindex
python -m evaluation.run_eval --llm-model llama3.1:8b          # if RAM allows
LLM_PROVIDER=openai python -m evaluation.run_eval --llm-model <model>   # OpenAI-compatible API
```

| Config | Model | Chunk size / overlap | Top-k | Retrieval hit rate | Answer accuracy | Groundedness | Abstention accuracy | Mean latency (s) | Results file |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | llama3.2:3b | 1000 / 200 | 4 | TBD | TBD | TBD | TBD | TBD | TBD |
| A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| C | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Discussion:** TBD — what changed between configurations, and why (relate to the failure classes).

## 7. Conclusions

TBD — fill in from the completed tables only.

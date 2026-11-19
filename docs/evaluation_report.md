# Evaluation Report: AI-Powered PDF Chatbot (RAG)

> **Status: DRAFT judgements; student confirmation required.**
> Every answer, retrieved chunk, latency and page reference quoted here comes from the real run files
> listed in §3. The *judgements* (correct / partial / incorrect, supported, failure class) were drafted by
> the reviewer and have **not yet been confirmed by the student**. Before submitting, check each row
> against the run file and the PDF, then change "Draft" to "Confirmed" (or correct the row).
> Rows marked **⚑ review** are ambiguous and need a human decision.

## 1. Methodology

**Goal.** Measure, separately, (a) whether retrieval puts the needed evidence into the LLM's context and
(b) whether the generated answer is correct, relevant, supported by that evidence, and abstains when
it should.

**Procedure.**
1. Questions were written against the actual PDFs. Every expected answer was verified in the extracted
   PDF text, with a verbatim quote and 1-based page recorded in `evaluation/questions*.json` (`notes`,
   `evidence_snippets`). Absence for unanswerable questions was verified by regex search over all pages.
2. `python -m evaluation.run_eval` ingests the corpus into a dedicated collection, asks every question
   through the same `RAGPipeline` the UI uses, and stores the retrieved chunks (rank, file, page, score,
   text), the answer, the sources shown, the abstention flag and the latency.
3. Each question was run at **top-k 4** (the default) and **top-k 8**.
4. Each answer was judged by reading the retrieved chunk text and comparing it with the verified expected
   answer. Semantically equivalent wording counts as correct. Retrieving the right *page* was not taken
   as proof that the right *passage* was retrieved.

**Automatic hints (not judgements).** The run files contain two hints:
- `expected_page_retrieved`: an expected (file, page) appears in the top-k.
- `evidence_retrieved` (V1.1): the verbatim evidence snippets appear in the retrieved text.

The snippet hint has **false negatives**. For example, M1's chunk says "BLEU score of 28.4", while the
snippet is "28.4 BLEU". The judgements below are therefore based on the chunk text, not on the hint.

**Metric definitions** (denominators are always stated):

| Metric | Population | Rule |
|---|---|---|
| Evidence retrieval | Answerable questions | **Full**: every fact the expected answer needs is in the retrieved chunks. **Partial**: some facts are. **None**: none are |
| Answer correctness | Answerable questions | **Correct**: all requested parts right (minor omissions of detail not asked for are allowed). **Partial**: some parts right, others missing or wrong. **Incorrect**: the main fact is wrong or absent (including an abstention when the answer exists) |
| Faithfulness | Every non-abstained answer | **Faithful** if every factual claim is supported by the retrieved chunks; an answer counts as unfaithful if it has even one unsupported or misattributed claim |
| Appropriate abstention | Unanswerable and false-premise questions | Unanswerable: the fallback was given. False premise: the system abstained or corrected the premise (abstaining is acceptable; correcting the premise is better) |

**Failure classes:**
- **Retrieval**: the needed evidence was not in the top-k.
- **Context construction**: the evidence was retrieved but lost or garbled before the LLM saw it (for example, a table flattened by text extraction).
- **Generation**: the evidence was available but the answer was wrong, incomplete or contradictory.
- **Abstention / hallucination**: the system answered something it should have declined, or declined something it could have answered.

## 2. System under test and dataset

| Item | Value |
|---|---|
| Code | Version 1 (commit `e04ea07`) for the baseline runs; **Version 1.1 (commit `8c322f4`)** for the final runs |
| Extraction / chunking | PyPDFLoader; RecursiveCharacterTextSplitter, chunk size 1000, overlap 200 |
| Embeddings / store | `sentence-transformers/all-MiniLM-L6-v2`, Chroma (cosine) |
| LLM | `llama3.2:3b` (Ollama, Q4_K_M), temperature 0 |
| Hardware | Apple M1, 8 GB RAM |
| Automated tests | `.venv/bin/pytest -q` → **110 passed** (V1.1); **123 passed** (V1.2, adds MMR and LLM-parameter tests) |

| Corpus | Documents | Pages | Chunks | Questions |
|---|---|---|---|---|
| **A** (`data/sample_pdfs/`) | Attention Is All You Need (1706.03762, 15 pp.), BERT (1810.04805, 16 pp.), Sentence-BERT (1908.10084, 11 pp.), RAG (2005.11401, 19 pp.) | 61 | 286 (V1.1) | 15 (`evaluation/questions.json`): 4 factual, 3 multi-passage, 2 synthesis, 1 partially answerable, 3 unanswerable, 2 false premise |
| **B** (`data/long_pdfs/`) | GPT-3, *Language Models are Few-Shot Learners* (2005.14165, 75 pp.); Llama 2 (2307.09288, 77 pp.) | 152 | 659 | 9 (`evaluation/questions_long.json`): 3 factual, 2 multi-passage (1 cross-document), 1 synthesis, 2 unanswerable, 1 false premise |

V1 used the 14 corpus A questions (P1 was added in V1.1). Corpus B was added in V1.1 so that two
sources longer than 70 pages are included (see §9 on the dataset-size question).

## 3. Run files

| Run | Code | Corpus | top-k | Questions | Mean latency |
|---|---|---|---|---|---|
| `results_20261011_113012` | V1 | A | 4 | 14 | 7.1 s |
| `results_20261011_113211` | V1 | A | 8 | 14 | 12.2 s |
| `results_20261011_121718` | **V1.1 (final)** | A | 4 | 15 | 7.2 s |
| `results_20261011_121920` | **V1.1 (final)** | A | 8 | 15 | 13.4 s |
| `results_20261011_122254` | **V1.1 (final)** | B | 4 | 9 | 6.7 s |
| `results_20261011_122408` | **V1.1 (final)** | B | 8 | 9 | 13.0 s |
| `results_20261119_222304` | V1.2 (MMR) | A | 4 (MMR) | 15 | 6.7 s |
| `results_20261119_222609` | V1.2 (MMR) | A | 8 (MMR) | 15 | 11.6 s |
| `results_20261119_222457` | V1.2 (MMR) | B | 4 (MMR) | 9 | 6.5 s |
| `results_20261119_222916` | V1.2 (MMR) | B | 8 (MMR) | 9 | 12.2 s |

The V1.2 runs use MMR retrieval (`--retrieval-mode mmr`, fetch_k 20, lambda 0.5). Everything else is
unchanged (same prompt, chunking, embeddings, LLM, temperature 0, context window 4096). They are
compared with the similarity runs in §8b.

Intermediate runs from partly fixed code are kept in `evaluation/results/superseded/` with an
explanation; they are not used for the figures below. V1's recorded `abstained` flag for A2 at k=4 is
`False` because of an abstention-detection defect that was later fixed (§7.3). The judgement below
treats that answer as the hedged abstention it is.

## 4. Summary of results (final V1.1 runs; draft)

### 4.1 Corpus A (15 questions)

| Metric | top-k 4 | top-k 8 |
|---|---|---|
| Evidence retrieval, full / partial / none (answerable, n=10) | 7 / 2 / 1 | 8 / 1 / 1 |
| Answers correct / partial / incorrect (answerable, n=10) | **7 / 2 / 1** | **6 / 3 / 1** |
| Faithful non-abstained answers | 9 / 10 | 10 / 11 |
| Unanswerable questions correctly abstained | 3 / 3 | 3 / 3 |
| False-premise questions handled appropriately | 2 / 2 (both abstained; premise not corrected) | 1 / 2 (A2 fabricated a rationale) |
| Mean latency | 7.2 s | 13.4 s |

### 4.2 Corpus B: long documents (9 questions)

| Metric | top-k 4 | top-k 8 |
|---|---|---|
| Evidence retrieval, full / partial / none (answerable, n=6) | 2 / 1 / 3 | 5 / 0 / 1 |
| Answers correct / partial / incorrect (answerable, n=6) | **2 / 1 / 3** | **3 / 1 / 2** |
| Faithful non-abstained answers | 4 / 4 | 3 / 4 |
| Unanswerable questions correctly abstained | 2 / 2 | 2 / 2 |
| False-premise question handled appropriately | 1 / 1 (abstained) | 1 / 1 (abstained) |
| Mean latency | 6.7 s | 13.0 s |

### 4.3 V1 baseline (corpus A, 14 questions; for comparison)

| Metric | V1 top-k 4 | V1 top-k 8 |
|---|---|---|
| Evidence retrieval, full / partial / none (answerable, n=9) | 6 / 2 / 1 | 6 / 2 / 1 |
| Answers correct / partial / incorrect (answerable, n=9) | 5 / 3 / 1 | 5 / 3 / 1 |
| Faithful non-abstained answers | 8 / 9 | 9 / 10 |
| Unanswerable correctly abstained | 3 / 3 | 3 / 3 |
| False premise handled appropriately | 2 / 2 | 1 / 2 |

**Reading these numbers.** The sample is small (10 and 6 answerable questions per corpus), so a change
of one question moves a rate by 10–17 percentage points. Differences between configurations should be
read as observations about specific questions, not as statistically reliable improvements.

## 5. Per-question judgements: corpus A, final V1.1 runs (draft)

Legend: Ev = evidence retrieval (Full / Partial / None); Outcome = Correct / Partial / Incorrect for
answerable questions; Abst. = correct abstention (N/A for answerable questions).

| ID | Category | k | Ev | Outcome | Relevant | Supported | Abst. | Failure class | Rationale (expected vs actual, evidence) |
|---|---|---|---|---|---|---|---|---|---|
| F1 | factual | 4 | Full | Correct | Y | Y | N/A | None | "6 identical layers [1] … d_model is 512 [2]"; matches p. 3 |
| F1 | factual | 8 | Full | Correct | Y | Y | N/A | None | Same answer |
| F2 | factual | 4 | Full | Correct | Y | Y | N/A | None | 110M / 340M, cited to the p. 3 chunk |
| F2 | factual | 8 | Full | Correct | Y | Y | N/A | None | Same |
| F3 | factual | 4 | None | Incorrect | Y | **N** | N/A | Retrieval + Generation | Expected: Dec 2018 Wikipedia dump, 21M documents (p. 4). The p. 4 chunk was not retrieved (it ranks 22nd; see §7.1). The answer cites the p. 19 memory-size passage, then draws an unsupported inference ("at least 15.3 billion values" as a document count) |
| F3 | factual | 8 | None | Incorrect | Y | Y | N/A | Retrieval | Same retrieval miss. It says the document count is not stated (true for its context), then appends the fallback sentence. No fabricated figure |
| F4 | factual | 4 | Full | Correct | Y | Y | N/A | None | 65 hours vs about 5 seconds; p. 1–2 (the snippet hint shows "partial" only because the chunk says "~5 seconds") |
| F4 | factual | 8 | Full | Correct | Y | Y | N/A | None | Same |
| M1 | multi_chunk | 4 | Full | Correct | Y | Y | N/A | None | 28.4 EN-DE, 41.0 EN-FR, 3.5 days on 8 P100. 41.0 is the figure in the p. 8 results text (the abstract and Table 2 say 41.8; the paper is inconsistent). The first sentence is uncited |
| M1 | multi_chunk | 8 | Full | Correct | Y | Y | N/A | None | Same answer |
| M2 | multi_chunk | 4 | Partial | Partial | Y | Y | N/A | Retrieval | The masking details (15%, 80/10/10) and the NSP 50/50 split were not retrieved. The model **correctly states** that masking details are missing. NSP is described from the retrieved text |
| M2 | multi_chunk | 8 | Partial | Partial | Y | Y | N/A | Generation (+ retrieval for NSP 50/50) | Masking now correct (15%, 80/10/10 [5]). NSP 50/50 never retrieved. The answer ends by claiming masking info could not be found, **contradicting itself** |
| M3 | multi_chunk | 4 | Full | Correct | Y | Y | N/A | None | Same-document vs per-token latent document; matches p. 2–3. The citation is a lone "[1]" at the end |
| M3 | multi_chunk | 8 | Full | Correct | Y | Y | N/A | None | Correct, but **no [n] markers**, so the Sources line falls back to all 6 retrieved pages |
| S1 | synthesis | 4 | Partial | Correct ⚑ review | Y | Y | N/A | None | Answers both parts: why raw BERT embeddings suit cosine poorly (SentEval passage, p. 7), and SBERT's siamese/triplet networks, 65 h → 5 s (p. 1). It omits the GloVe comparison and the 54.81/29.19 figures, which the question did not explicitly request. Decide whether that omission makes it partial |
| S1 | synthesis | 8 | Full | Partial | Y | Y | N/A | Generation | The "how SBERT addresses it" part mentions only the pooling layer, not the siamese/triplet fine-tuning, even though the p. 3 chunk with that text was retrieved |
| S2 | synthesis | 4 | Full | Partial | Y | Y | N/A | Generation | BERT-base query and document encoders of DPR: correct. The generator is described as "pθ" and **never named as BART-large**, although the p. 3 chunk containing "BART-large" was retrieved. No [n] markers, so Sources shows all 4 retrieved pages |
| S2 | synthesis | 8 | Full | Partial | Y | Y | N/A | Generation | Adds "110M parameters each" (p. 18, correct); still does not name BART-large |
| P1 | partial (adversarial) | 4 | Full | Correct | Y | Y | N/A | None | "6 identical layers [1]", then says the dollar cost cannot be found. Exactly the required partial-answer behaviour |
| P1 | partial (adversarial) | 8 | Full | Correct | Y | Y | N/A | None | Same |
| U1 | unanswerable | 4 / 8 | N/A | N/A | Y | N/A | **Yes** | None | Exact fallback; GPT-4 is absent from the corpus |
| U2 | unanswerable | 4 / 8 | N/A | N/A | Y | N/A | **Yes** | None | Exact fallback; MMLU is absent |
| U3 | unanswerable | 4 / 8 | N/A | N/A | Y | N/A | **Yes** | None | Exact fallback; no CO2 figures. It did not misuse the "3.5 days on 8 P100" context |
| A1 | false premise | 4 | None | N/A | Y | N/A | Yes (acceptable) | None | Abstained. The premise ("LSTM layers") was not corrected; the "dispensing with recurrence" chunk (p. 1–2) was not retrieved |
| A1 | false premise | 8 | None | N/A | Y | N/A | Yes (acceptable) | None | Same |
| A2 | false premise | 4 | None | N/A | Y | N/A | Yes (acceptable) | None | Abstained (fixed by V1.1 prompt rule 10; see §8). The "default configuration is MEAN" chunk (p. 3) was not retrieved |
| A2 | false premise | 8 | None | N/A | Y | **N** | **No** | Abstention / hallucination (+ retrieval) | **Accepted the false premise** and built a rationale from BERT's [CLS] fine-tuning passage (BERT p. 5). The MEAN-pooling chunk was never retrieved. Rule 10 did not prevent this at k=8 |

## 5b. Per-question judgements: corpus A, V1 baseline runs (draft)

V1 had no P1. "Same as V1.1" means the V1 answer is substantively the same as the final run's answer
for that k, with the same judgement.

| ID | k=4 outcome (Ev) | k=8 outcome (Ev) | Failure class | Rationale |
|---|---|---|---|---|
| F1 | Correct (Full) | Correct (Full) | None | Same as V1.1 |
| F2 | Correct (Full) | Correct (Full) | None | Same as V1.1 (V1 cited [3] for the same p. 3 chunk) |
| F3 | Incorrect (None) | Incorrect (None) | Retrieval; at k=4 also Generation | k=4 invents "at least 5 documents" from a passage about the number of retrieved documents (unsupported). k=8 says the count is not specified |
| F4 | Correct (Full) | Correct (Full) | None | 65 h vs ~5 s |
| M1 | Correct (Full) | Correct (Full) | None | 28.4 / 41.0 (p. 8 text) / 3.5 days on 8 P100 |
| M2 | Partial (Partial) | Partial (Partial) | k=4 Retrieval + Generation; k=8 Generation | k=4: generic masking description, with no note that the specifics are missing. k=8: 80/10/10 correct but omits 15% (it was retrieved) and NSP 50/50, and ends with a contradictory fallback sentence |
| M3 | Correct (Full) | Correct (Full) | None | Correct definitions; citation placed after the paragraph |
| S1 | Partial (Partial) | Partial (Partial) | Generation | Explains only why raw BERT embeddings are poor; never says how SBERT addresses it, and does not state that this part is missing |
| S2 | Partial (Full) | Partial (Full) | Generation | Correct DPR / BERT-base encoders; generator not named as BART-large although it was retrieved |
| U1–U3 | Correct abstention | Correct abstention | None | Exact fallback in all 6 responses |
| A1 | Acceptable abstention | Acceptable abstention | None | Premise not corrected |
| A2 | Acceptable abstention (hedged) | **Fabricated rationale** | k=8 Abstention / hallucination (+ retrieval) | k=4: "The passage does not explicitly state why…" followed by the fallback (recorded `abstained=False` due to the V1 detection defect). k=8: accepted the premise and justified it from BERT [CLS] passages |

## 6. Per-question judgements: corpus B (long documents), final V1.1 runs (draft)

| ID | Category | k | Ev | Outcome | Relevant | Supported | Abst. | Failure class | Rationale |
|---|---|---|---|---|---|---|---|---|---|
| LF1 | factual | 4 | None | Incorrect | Y | N/A | No (answer existed) | Retrieval | The Table 2.1 / n_ctx chunk (p. 8) was not retrieved, so the model abstained. That is the right behaviour for its context, but no answer was given |
| LF1 | factual | 8 | Full | Partial | Y | **N** | N/A | Generation | 175B parameters and 96 layers are correct (cited to the p. 8 chunk, which also contains "n_ctx = 2048 tokens"). It then claims "a context window of 32 examples", **misattributing** the SuperGLUE few-shot setting (p. 18) |
| LF2 | factual | 4 | Full | Correct | Y | Y | N/A | None | 410 billion tokens, 60% weight (p. 9). This was a flattened table, and the model read it correctly |
| LF2 | factual | 8 | Full | Correct | Y | Y | N/A | None | Same |
| LF3 | factual | 4 | None | Incorrect | Y | Y | N/A | Retrieval | The p. 26 "barely above chance at ~52%" chunk was not retrieved (another p. 26 chunk was). The answer says the information is not provided. No fabricated number |
| LF3 | factual | 8 | Full | Incorrect | Y | N/A | **No (wrongly abstained)** | Abstention (generation) | The evidence chunk **was** retrieved, yet the model gave the fallback: an over-cautious abstention |
| LM1 | multi_chunk | 4 | Full | Correct | Y | Y | N/A | None | 2T tokens, 3.3M GPU hours on A100-80GB, 539 tCO2eq, 100% offset (pp. 7, 77) |
| LM1 | multi_chunk | 8 | Full | Correct | Y | Y | N/A | None | Same |
| LM2 | multi_chunk (cross-doc) | 4 | None | Incorrect | Y | N/A | No (answer existed) | Retrieval | Neither the GPT-3 n_ctx chunk nor the Llama 2 "2048 → 4096" chunk was retrieved, so the model abstained |
| LM2 | multi_chunk (cross-doc) | 8 | None | Incorrect | Y | N/A | No (answer existed) | Retrieval | Same. The comparative wording does not match either source passage well |
| LS1 | synthesis | 4 | Partial | Partial | Y | Y | N/A | Retrieval (+ generation) | Rejection sampling + PPO, correct [p. 5]. Ghost Attention not retrieved. Adds an unrequested remark about reward signals |
| LS1 | synthesis | 8 | Full | Correct | Y | Y | N/A | None | Rejection sampling + PPO, GAtt for multi-turn consistency, and safety improvements, all cited (pp. 5, 8, 31) |
| LU1 | unanswerable | 4 / 8 | N/A | N/A | Y | N/A | **Yes** | None | Exact fallback; it did not use the $12 participant payments |
| LU2 | unanswerable | 4 / 8 | N/A | N/A | Y | N/A | **Yes** | None | Exact fallback; it did not substitute plain MMLU scores |
| LA1 | false premise | 4 / 8 | None | N/A | Y | N/A | Yes (acceptable) | None | Abstained; the "does not include data from Meta's products" chunk (p. 5) was not retrieved, so the premise was not corrected |

## 7. Known-defect investigations

### 7.1 F3: retrieval miss (unresolved; root cause identified)

- **Evidence.** The text is on RAG p. 4: "we use the December 2018 dump. Each Wikipedia article is split into
  disjoint 100-word chunks, to make a total of 21M documents." It is present and clean in the index
  (chunk `…-p4-c?`, 954 characters).
- **Ranking.** For the F3 wording, this chunk ranks **22nd** (cosine 0.481). The top results are p. 18–19
  passages about "non-parametric memory" and "index", which match the question's vocabulary more
  closely. The same chunk ranks **1st** for "What Wikipedia dump is used and how many documents are in
  the index?" and **3rd** for "How many documents does the RAG Wikipedia index contain?".
- **Contributing factor.** About 40% of the chunk is the end of the previous section ("Fast Decoding"
  beam-search text), which dilutes its embedding.
- **Root cause.** A wording mismatch between the question and the passage: the question uses the
  abstract's terms, the passage uses "Wikipedia dump". Mixed-topic chunk boundaries make it worse.
  This is a dense-retrieval limitation, not a code defect.
- **Experiments (retrieval only, no code change):**
  - The chunk-size sweep (1000/200, 800/160, 600/120, 500/100 at k = 4 and 8) never retrieved F3. The
    other per-question differences moved in both directions, so the default was kept.
  - Ligature normalisation (V1.1) raised the score from 0.473 to 0.481 without changing the rank.
  - Swapping in `BAAI/bge-small-en-v1.5` (same 384 dimensions) moved the chunk from rank 22 to rank 13.
    That is still outside the top 8, and the change lost F2's evidence, so the default model was kept.
- **Status.** Not fixed. A reranker or hybrid BM25 + dense retrieval is the evidence-based next step
  (the exact term "Wikipedia" would be matched by BM25). Neither was implemented, per the brief's
  "no complex retrieval without evidence" guidance; this evaluation now provides that evidence.

### 7.2 Incorrect BLEU value 38.1 vs 28.4 (mitigated; verified on a fixed set of phrasings)

- **Root cause.** Text extraction flattens Table 2 (Transformer p. 8) into rows of bare numbers. Some
  rows have missing cells, so the columns don't line up, and the EN-DE / EN-FR headers are separated
  from the values. The row "Transformer (base model) 27.3 38.1 3.3·10^18" makes 38.1 look like a
  Transformer score; it is actually the **base** model's **EN-FR** value.
- **Reproduction.** Before the fix, "What is the English-to-German BLEU score of the Transformer according
  to Table 2?" answered "38.1 … 41.8": wrong values from the wrong column.
- **Classification.** Context construction (table structure lost; table extraction is out of scope for the
  baseline) plus generation (the model picked a number out of an ambiguous row).
- **Fix.** Prompt rule 9: copy numbers exactly; prefer a sentence that states the number; take a number
  from a table row only when its row and column are clear, otherwise say the table is ambiguous.
  Regression test: `tests/test_generation.py::test_system_prompt_has_number_and_table_rule` (unit
  level: it checks the rule is present, not model behaviour).
- **Real-model verification.** 8 BLEU phrasings, one run each, `llama3.2:3b`, temperature 0. After the
  fix, none produced a wrong number. The Table 2 phrasing now **abstains** instead (safe, but an
  over-cautious answer, since 28.4 is in the documents). Two "EN-DE" abbreviation phrasings also
  abstained both before and after the fix. M1 in the final runs reports 28.4 correctly at both k values.
- **Status.** Mitigated on the tested phrasings, **not proven eliminated**. A 3B model can still misread
  numbers in other tables. LF2 (a flattened GPT-3 table) was read correctly.

### 7.3 Other issues found by the audit and in this evaluation

| Issue | Verified? | Action | Regression coverage |
|---|---|---|---|
| Sources line listed every retrieved page, not the cited ones | Yes | Fixed: `cited_sources()` uses `[n]` markers and falls back to all pages if there are none | `tests/test_pipeline.py` (3 tests) |
| Hedged fallback (A2, V1 k=4) not flagged as an abstention | Yes | Fixed; the first fix was too broad (it mislabelled M2), so it was **narrowed** to hedges of ≤ 30 words | `tests/test_pipeline.py::test_is_abstention_true/false` |
| Ligatures (`ﬁ`) split words for the tokenizer; hyphen re-join merged real compounds | Yes | Fixed in `normalize_text` | `tests/test_pdf_loader.py::test_normalize_text` (3 new cases) |
| A2 fabricated a rationale for a false premise (V1 k=8; V1.1 k=4 and k=8) | Yes | Prompt rule 10. Verified: fixed at k=4, **still fails at k=8** | `test_system_prompt_has_false_premise_rule` (unit level) |
| Duplicate check ran after the full parse | Yes | Moved before parsing | `tests/test_ingest.py::test_duplicate_detected_before_parsing` |
| Ollama "model not found" surfaced as a raw error | Yes | Mapped to `LLMInitError` with a pull hint | `tests/test_generation.py::test_ollama_response_error_wrapped_with_pull_hint` |
| A second store instance broke after another instance's reset | Yes | Reopen on `NotFoundError` | `tests/test_vectorstore.py::test_second_instance_survives_reset_by_another` |
| Dependencies unpinned | Yes | Not changed; recommended (`pip freeze > requirements-lock.txt`) | n/a |

## 8. Top-k 4 vs top-k 8 (observations, not a general conclusion)

- **More evidence retrieved at k=8.** Corpus A: full evidence 7 → 8 of 10 (S1). Corpus B: 2 → 5 of 6
  (LF1, LF3, plus LS1 from partial to full). M2's masking details appear only at k=8.
- **But not better answers across the board.**
  - Corpus A correct answers went 7 → 6, because S1 became partial.
  - A2 produced a hallucination only at k=8, since the extra BERT [CLS] passages gave the model material
    for a plausible-sounding false rationale.
  - In corpus B, LF1 gained evidence but answered with a misattributed fact, and LF3 gained evidence but
    abstained anyway.
- **Cost.** Mean latency roughly doubled (7.2 s → 13.4 s on corpus A; 6.7 s → 13.0 s on corpus B).
- **Conclusion for this system.** k=8 improves recall, especially on the long documents, but the 3B model
  does not reliably exploit the larger context, and the extra context sometimes hurts. k=4 stays the
  default. Retrieval quality (reranking) looks more promising than simply raising k. This is based on
  15 + 9 questions and one run per configuration.

This is a retrieval-configuration comparison, **not** an LLM comparison. No second LLM was evaluated.

## 8b. Similarity vs MMR retrieval (V1.2; draft judgements)

**What MMR does.** Maximal Marginal Relevance fetches the 20 most similar chunks, then picks k of them
one at a time, scoring each candidate as `0.5 × similarity to the question − 0.5 × similarity to the
chunks already picked`. The aim is less redundancy, so more distinct passages reach the LLM.

### Summary (answerable questions; same judging rules as §1)

| Corpus, k | Search | Evidence full / partial / none | Correct / partial / incorrect | Faithful | False premise OK | Latency |
|---|---|---|---|---|---|---|
| A, 4 | Similarity | 7 / 2 / 1 | **7 / 2 / 1** | 9 / 10 | 2 / 2 | 7.2 s |
| A, 4 | MMR | 4 / 4 / 2 | **4 / 3 / 3** | 8 / 9 | 2 / 2 | 6.7 s |
| A, 8 | Similarity | 8 / 1 / 1 | **6 / 3 / 1** | 10 / 11 | 1 / 2 | 13.4 s |
| A, 8 | MMR | 4 / 5 / 1 | **4 / 5 / 1** | 9 / 10 | **2 / 2** | 11.6 s |
| B, 4 | Similarity | 2 / 1 / 3 | **2 / 1 / 3** | 4 / 4 | 1 / 1 | 6.7 s |
| B, 4 | MMR | 3 / 3 / 0 | **3 / 2 / 1** | 4 / 5 | 1 / 1 | 6.5 s |
| B, 8 | Similarity | 5 / 0 / 1 | **3 / 1 / 2** | 3 / 4 | 1 / 1 | 13.0 s |
| B, 8 | MMR | 4 / 2 / 0 | **3 / 3 / 0** | 4 / 6 | 1 / 1 | 12.2 s |

All 5 unanswerable questions still abstained with MMR at both k values (10 of 10 responses).

### Per-question changes (only questions whose judgement changed)

| ID | k | Similarity | MMR | Why (from the chunk text) |
|---|---|---|---|---|
| F3 | 4 | Incorrect | Incorrect | Still a retrieval miss. Now **unfaithful in a new way**: calls BART-large (the generator) the "knowledge source" |
| F3 | 8 | Incorrect | Partial ⚑ review | First time F3 gets anything: "Wikipedia … 21M … vectors" from the p. 19 memory passage (21M 728-dim vectors). The p. 4 dump passage is still not retrieved, so "December 2018" is missing |
| F4 | 4 | Correct | **Incorrect** | MMR kept the p. 1 "65 hours" chunk but dropped the p. 2 / p. 8 chunks with "~5 seconds", which are close in meaning to the p. 1 chunk. The model then abstained, even though the BERT half was in its context |
| F4 | 8 | Correct | Partial | Same loss of the "5 seconds" chunks; this time it gives the 65 hours and says the SBERT figure is missing |
| M2 | 4 / 8 | Partial | Partial | Unchanged outcome. At k=8 the self-contradicting trailing fallback has gone; 15% is retrieved but still omitted |
| M3 | 4 / 8 | Correct | **Incorrect** | The definition is split over two **neighbouring** chunks on p. 3 (200-character overlap). MMR treats the second one as redundant with the first and drops it, so the "same document vs. different document per token" text never reaches the model |
| S1 | 4 | Correct ⚑ | Partial | The "how SBERT fixes it" part is vague ("a different setup"); the siamese/triplet passage was not retrieved |
| A2 | 8 | **Fabricated rationale** | **Abstained** | With fewer near-identical BERT [CLS] passages in the context, the model no longer builds a false rationale. This is the only false-premise hallucination in the final runs, and MMR removes it |
| LF1 | 4 | Incorrect (abstained) | Partial | MMR brings in the p. 8 Table 2.1 chunk: 96 layers correct. But "12288 parameters" (that is d_model) and "context window of one Reuters article" are misreadings, so **unfaithful** |
| LF3 | 4 | Incorrect | **Correct** | MMR reaches the p. 27 Table 3.12 chunk: 52% vs 88% for the control model |
| LF3 | 8 | Incorrect (wrongly abstained) | **Correct** | Same passage; this time the model uses it |
| LM2 | 8 | Incorrect (abstained) | Partial ⚑ review | The cross-document question now gets an answer: "GPT-3 2048, Llama 2 4096". Both numbers are right, but the GPT-3 figure is cited to the Llama 2 sentence "from 2048 [Llama 1] to 4096 tokens". The GPT-3 n_ctx chunk was not retrieved, so the GPT-3 part is **right by coincidence and unfaithful** |
| LS1 | 4 | Partial | Partial | Swapped gaps: similarity had rejection sampling + PPO but no Ghost Attention; MMR has Ghost Attention but not rejection sampling + PPO |
| LS1 | 8 | Correct | Partial | Rejection sampling, PPO and Ghost Attention are all missing from the answer, although the GAtt passage (p. 8) was retrieved |

Other details: F1 at k=8 with MMR is correct, but its citations came out as section titles
("[3.1 Encoder and Decoder Stacks]") instead of [n], so Sources falls back to all retrieved pages.
LF2 at k=4 adds an unrequested, muddled remark about WebText2's weight.

### Observations

- **MMR hurts on the short-paper corpus and helps on the long one.** Corpus A: correct answers 7 → 4
  (k=4) and 6 → 4 (k=8). Corpus B: incorrect answers 3 → 1 (k=4) and 2 → 0 (k=8).
- **Root cause of the losses: chunk overlap.** With 1000/200 chunking, the continuation of a passage is
  in the neighbouring chunk, and the two chunks are similar. MMR reads that similarity as redundancy and
  drops the neighbour (M3, F4). In the short papers, the needed facts are often spread across such
  neighbours.
- **Why it helps on long documents.** A 75–77-page paper has many near-duplicate passages on the same
  topic (abstract, introduction, results, appendix). Similarity search fills k with these; MMR spreads
  the k slots across more sections (LF3, LF1 and LM2 each reached a passage similarity search missed).
- **Less redundancy reduced one hallucination but added misattributions.** A2's fabricated rationale
  disappeared at k=8, but LF1 (k=4) and LM2 (k=8) built answers from loosely related numbers. More
  diverse context gives a 3B model more unrelated numbers to confuse.
- **Decision.** **Similarity stays the default** (it is better on corpus A, and the default is what most
  users of a few short PDFs will hit). MMR is available as a setting in the UI and in `RETRIEVAL_MODE`.
  It is the better choice for long reports, where the evidence is spread across many sections. A sweep
  of `MMR_LAMBDA` (for example 0.7) and of smaller chunk overlap with MMR was **not** done and would be
  the next experiment.
- Same caveats as §8: 10 + 6 answerable questions, one run per setting, draft judgements.

## 8c. LLM parameters (V1.2)

The generation settings are now explicit and visible in the UI:

| Parameter | Value | Why |
|---|---|---|
| Model | `llama3.2:3b` via Ollama | Runs on an 8 GB laptop; no API cost; data stays local |
| Temperature | 0.0 | Deterministic, most literal use of the context. Grounded QA needs faithfulness, not variety |
| Top-p | 0.9 | Ollama's default. With temperature 0 decoding is greedy, so top-p has no effect; it matters only if temperature is raised |
| Context window (`num_ctx`) | 4096 tokens | Measured: Ollama 0.17.4 cuts prompts to 4096 tokens by default on this machine. The largest prompt in any run (S1, top-k 8) is **2,545 tokens**, so prompt and answer fit and no evaluated run was truncated. Raise it for chunk sizes above 1000 or top-k above 10 |
| Retrieval top-k | 4 (UI 1–10) | §8 |
| Retrieval mode | similarity (MMR optional) | §8b |
| Chunk size / overlap | 1000 / 200 characters | §7.1 sweep |

## 9. Limitations of this evaluation

- Small sample: 10 + 6 answerable questions, one run per configuration. Temperature 0 makes runs largely
  repeatable, but small prompt changes did alter individual answers (see `results/superseded/`).
- One evaluator drafted all judgements; they require the student's confirmation (see the status banner).
- Both corpora are machine-learning papers; behaviour on other domains was not tested.
- Expected answers for multi-part questions involve judgement about which details are "required".
  Those cases are flagged ⚑.
- **Dataset size requirement.** The brief says about 50–200 pages without stating whether that applies
  per document or per collection. Corpus A is 61 pages (documents of 11–19 pp.). Corpus B is 152 pages
  (75 + 77 pp.). **To confirm with the professor.**

## 10. Conclusions

1. **Abstention on unanswerable questions is reliable in these tests.** All 5 unanswerable questions
   abstained at both k values in V1.1 (10 of 10 responses), without using tempting near-miss context.
2. **Retrieval is the dominant failure source.** Of the incorrect or partial answers in the final runs,
   F3, LF1 (k=4), LF3 (k=4), LM2, LS1 (k=4) and M2's missing details trace to evidence that was not
   retrieved. Cross-document comparisons (LM2) and passages whose wording differs from the question (F3)
   are the weak spots. More of these misses occur on the long documents.
3. **Generation failures are real but rarer.** Specifically:
   - details left out even though they were retrieved (S2's BART-large, S1 at k=8)
   - one misattributed fact (LF1 at k=8)
   - self-contradictory trailing fallbacks (M2 at k=8, F3 at k=8)
   - one over-cautious abstention (LF3 at k=8)
4. **False premises remain the main hallucination risk.** The system never corrected a premise. It
   abstained in 5 of 6 false-premise responses and fabricated once (A2, k=8).
5. **MMR (V1.2) is corpus-dependent.** It helped on the long documents (no incorrect answers at k=8,
   LF3 fixed at both k) and removed the A2 hallucination, but hurt on the short papers (M3, F4) by
   dropping neighbouring chunks. Similarity stays the default (§8b).
6. **V1.1 fixes:**
   - **Measured on these questions:** A2 at k=4 (fabrication → abstention), S1 at k=4 (partial →
     correct), citations limited to cited pages, P1 partial-answer behaviour.
   - **Not demonstrated:** a general accuracy gain.

## 11. Unresolved issues and suggested improvements

| Issue | Evidence | Suggested improvement (not implemented) |
|---|---|---|
| Wording-mismatch retrieval misses | F3, LF3 (k=4), LM2, A2/A1/LA1 premise chunks | Hybrid BM25 + dense retrieval, or a cross-encoder reranker over the top 20. (MMR, tried in §8b, helps on the long corpus but does not fix F3 and hurts the short-paper corpus) |
| MMR drops neighbouring chunks that continue a passage | M3, F4 (§8b) | Merge adjacent chunks after retrieval, or tune `MMR_LAMBDA` / chunk overlap |
| Cross-document comparisons | LM2 | Query decomposition (one sub-query per document) |
| False-premise handling | A2 (k=8) | Stronger LLM, or a premise-check step |
| Omitted retrieved details; contradictions | S2, M2 (k=8) | Stronger LLM (OpenAI-compatible path is already supported), or answer/claim verification |
| Table numbers | BLEU case | Table-aware extraction (out of scope for baseline) |

## 12. How to reproduce

```bash
.venv/bin/pytest -q
.venv/bin/python -m evaluation.run_eval --questions evaluation/questions.json --top-k 4 --reindex
.venv/bin/python -m evaluation.run_eval --questions evaluation/questions.json --top-k 8
.venv/bin/python -m evaluation.run_eval --questions evaluation/questions_long.json --pdf-dir data/long_pdfs --top-k 4 --reindex
.venv/bin/python -m evaluation.run_eval --questions evaluation/questions_long.json --pdf-dir data/long_pdfs --top-k 8
# V1.2: MMR retrieval (repeat with --top-k 8, and with the long corpus)
.venv/bin/python -m evaluation.run_eval --questions evaluation/questions.json --top-k 4 --retrieval-mode mmr
```

New runs write new timestamped files. With temperature 0, individual answers are usually but not always
identical to those quoted here.

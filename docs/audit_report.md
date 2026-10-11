# Independent Audit Report — PDF RAG Chatbot

> **Auditor draft — student must verify before submission.**
> Everything below was checked by reading the code and running it on 2026-10-11 (macOS, `.venv`,
> Ollama `llama3.2:3b`). The evaluation assessment in Section 6 is the auditor's own opinion,
> not the student's human judgement. It does **not** replace `docs/evaluation_template.md`, which is
> still unfilled (all `TBD`). Re-check each verdict against the results files before you copy anything.

## 1. Summary verdict

**The build is substantially compliant and works.** The app starts, all 98 tests pass, ingestion
rejects bad files with clear messages, retrieval is limited to the selected corpus, and the exact
fallback sentence is returned for the three unanswerable questions in both evaluation runs. No
results are fabricated: every human-judgement field in both runs is blank, and the evaluation
report is still a template.

What still needs work:

1. **The manual evaluation has not been done.** `docs/evaluation_template.md` is all `TBD`. This is
   the main deliverable still open.
2. **The Sources line is too broad.** It lists every retrieved page, not just the passages the
   answer cites. An answer that hedges before giving the fallback sentence is not flagged as
   abstained, so it still shows sources.
3. **Text normalization has two small defects.** Ligatures (`ﬁ`, `ﬂ`) are kept, and line-break
   hyphens inside real compounds are removed (`English-to-German` becomes `Englishto-German`).
   Both hurt embedding quality.
4. **Observed answer quality is mixed.** Retrieval misses (F3, M2, A2) and a 3B model that leaves
   out requested parts without saying so (S1, S2). Once one run accepted a false premise (A2 at
   top-k 8).

Compliance matrix: **57 PASS, 5 PARTIAL, 0 FAIL** out of 62 checked items (Section 3). Every
PARTIAL is listed in Section 3 and again in Section 7.

## 2. Test results (observed)

```
$ .venv/bin/pytest -q
........................................................................ [ 73%]
..........................                                               [100%]
98 passed in 14.51s

$ .venv/bin/pytest -q -m "not slow"
97 passed, 1 deselected in 2.75s
```

After the auditor's fixes (Section 8) the full suite was run again: `98 passed in 14.95s`.

Coverage against the brief: PDF validation and loading (`tests/test_pdf_loader.py`, 18 tests),
chunk metadata (`tests/test_chunking.py`), indexing, dedupe, reset and filtering
(`tests/test_vectorstore.py`, `tests/test_ingest.py`), known-answer retrieval
(`tests/test_retrieval.py::test_known_facts_retrieved_on_right_page`, plus a real-model semantic
test marked `slow`), empty and invalid input (blank question, empty bytes, wrong extension,
renamed non-PDF, corrupted, encrypted, blank PDF), unanswerable handling
(`test_no_chunks_abstains_without_llm`, `test_model_abstention_clears_sources`,
`test_min_relevance_dropping_everything_falls_back`), and end-to-end with mocks
(`tests/test_pipeline.py::test_end_to_end_with_real_store`, `tests/test_app_smoke.py` with AppTest).

## 3. Requirement-by-requirement compliance matrix

### A. Ingestion

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| A1 | Multi-PDF upload | PASS | `app.py:197-204` (`accept_multiple_files=True`); `rag/ingest.py` handles each file on its own |
| A2 | Unsupported formats rejected with a clear error | PASS | `rag/pdf_loader.py:54-69` (extension + `%PDF-` magic); uploader `type=["pdf"]` `app.py:198`; live check §5.3 |
| A3 | Text extraction | PASS | `rag/pdf_loader.py:129` (`PyPDFLoader`) |
| A4 | Filename + page metadata | PASS | `rag/pdf_loader.py:144,149` (1-based `page`, basename `source`, `file_hash`); kept on every chunk `rag/chunking.py:37-39` |
| A5 | Empty pages | PASS | `rag/pdf_loader.py:146-147,163-167` (skipped and reported as a warning) |
| A6 | Unreadable or corrupted PDFs | PASS | `rag/pdf_loader.py:90-111,128-134`; live: truncated PDF and random bytes with a `%PDF-` header both rejected |
| A7 | Duplicates | PASS | `rag/ingest.py:50-54` (SHA-256; a copy under another name is detected live). Note: the check runs *after* the full parse, so a duplicate costs one extraction |
| A8 | No-text PDFs; no OCR, with an explanation | PASS | `rag/pdf_loader.py:156-160`; live: "probably a scanned or image-only PDF; OCR is not supported" |
| A9 | Normalization without losing meaning | **PARTIAL** | `rag/pdf_loader.py:72-87`. (a) Ligatures are not mapped: chunks contain `ﬁxed` / `ﬁne-tuning`, and the MiniLM tokenizer splits them into `['ﬁ', '##xed']` (verified). (b) `_HYPHEN_BREAK` also joins real compounds split at a line break: the indexed chunk reads "Englishto-German translation" (1706.03762 p.1). Meaning is mostly kept, but retrieval quality suffers |
| A10 | Dataset size and confidentiality | PASS | 4 public arXiv PDFs, 61 pages in total (15/16/11/19), verified with pypdf |

### B. Chunking

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| B1 | Overlapping chunks, LangChain splitter | PASS | `rag/chunking.py:21-26` (`RecursiveCharacterTextSplitter`); `test_consecutive_chunks_overlap` |
| B2 | Size and overlap configurable | PASS | `rag/config.py:47-48,79-82`; UI `app.py:107-114`; CLI `--chunk-size/--chunk-overlap` |
| B3 | Metadata preserved | PASS | `rag/chunking.py:37-39` (+`chunk_index`, `chunk_id`) |
| B4 | Config recorded for evaluation | PASS | `evaluation/run_eval.py:378` (`settings.as_report_dict()`); appears in both results files; separate collection per chunk config `run_eval.py:44-49` |

### C. Embeddings

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| C1 | Pretrained model | PASS | `rag/embeddings.py:33-37` (`all-MiniLM-L6-v2`, normalized) |
| C2 | Same model for queries | PASS | the Chroma `embedding_function` embeds both documents and queries, `rag/vectorstore.py:44-50` |
| C3 | Configurable | PASS | `EMBEDDING_MODEL`, `rag/config.py:51-53` |
| C4 | Clear errors on init or download failure | PASS | `rag/embeddings.py:22-46`; live: `EmbeddingInitError: Failed to load embedding model 'nonexistent-org/no-such-model-xyz': ... Check that (1) you have an internet connection ...`; the UI stops with a message (`app.py:139-146`) |

### D. Vector DB

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| D1 | Chunks, embeddings and metadata stored | PASS | `rag/vectorstore.py:57-75` (ids = `chunk_id`) |
| D2 | Semantic (dense) search | PASS | `rag/vectorstore.py:117-135` (cosine HNSW); real-model paraphrase test passes |
| D3 | Configurable top-k | PASS | `TOP_K`; UI slider `app.py:106`; `--top-k` |
| D4 | Retrieval limited to the active corpus | PASS | `rag/vectorstore.py:126-133` (`file_hash $in`); `app.py:372` passes the ticked files; live Q4 (§5.4) |
| D5 | Reset / re-index without mixing documents | PASS | `rag/vectorstore.py:77-80` (delete and recreate the collection); `app.py:211-223,278-296`. Caveat: if chunk settings change and new files are added *without* re-indexing, chunk configs mix within one collection (the UI warns about this, `app.py:115-118`) |
| D6 | Local store | PASS | persistent Chroma in `./chroma_db` (gitignored) |

### E. Generation

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| E1 | Question + retrieved context sent to the LLM | PASS | `rag/generation.py:127-147` |
| E2 | Explicit system prompt restricting answers to the context | PASS | `rag/generation.py:28-48` |
| E3 | No silent use of general knowledge | **PARTIAL** | Enforced by the prompt only; there is no check of the answer afterwards. Observed: A2 (top-k 8) invents an SBERT rationale from BERT `[CLS]` passages; F3 (top-k 4) makes an unsupported inference ("at least 5 documents") |
| E4 | Exact fallback sentence | PASS | `rag/config.py:19-21`; returned word for word for U1–U3 in both runs and in live checks; LLM skipped when nothing is retrieved (`rag/pipeline.py:95-98`) |
| E5 | Partial answers say what is unsupported | **PARTIAL** | Prompt rule 4 (`rag/generation.py:36`). Worked in 2/2 live partial questions (§5.4). But S1 and S2 in both runs left out requested parts without saying so, and M2 (top-k 8) added the full fallback sentence after a substantive answer |
| E6 | Source filename + page for substantive answers | **PARTIAL** | Shown (`app.py:329-331`), but `sources` = *all* retrieved pages, not the cited ones (`rag/pipeline.py:51-60,107`). Example: F2 lists `arxiv_2005.11401.pdf p.18` although only [3] (BERT p.3) is cited. Also, `is_abstention` only checks the *start* of the answer (`rag/pipeline.py:39-48`), so A2 (top-k 4), "The passage does not explicitly state ... I could not find enough information ...", gets `abstained=False` and sources are displayed for a non-answer |
| E7 | Provider configurable | PASS | `rag/generation.py:58-96` (ollama / openai-compatible) |
| E8 | No hardcoded credentials | PASS | key read from env only (`rag/generation.py:78`); grep found no secrets (§5.6) |

### F. UI (Streamlit)

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| F1 | Multi-upload | PASS | `app.py:197-200` |
| F2 | Indexing status | PASS | `st.status` + progress bar, `app.py:151-175` |
| F3 | Success / failure feedback | PASS | `app.py:179-189` (success / info for duplicate / error) |
| F4 | Chat | PASS | `app.py:359-388`; live AppTest (§5.2) |
| F5 | Question and answer shown | PASS | `app.py:345-350,364-387` |
| F6 | Source references | PASS | `app.py:329-331` (but see E6 on precision) |
| F7 | Inspect retrieved evidence | PASS | `app.py:333-342` (rank, file, page, similarity, chunk_id, text) |
| F8 | Clear / re-index corpus | PASS | `app.py:268-296`; covered by `test_deselect_remove_and_clear`. Re-index only works for files uploaded in the current session (the UI warns) |

### Code quality, testing, hygiene

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| Q1 | Modular layout | PASS | `rag/{config,pdf_loader,chunking,embeddings,vectorstore,ingest,retrieval,generation,pipeline}.py`, `app.py`, `evaluation/run_eval.py` |
| Q2 | Pytest coverage of the required areas | PASS | §2 |
| Q3 | `.gitignore` | PASS | excludes `.venv`, `.env`, `chroma_db`, caches. It used to also ignore results JSON (fixed, §8) |
| Q4 | Dependencies file | PASS | `requirements.txt` (unpinned `>=` versions; see risks) |
| Q5 | `.env.example` | PASS | present, no secrets |
| Q6 | No secrets in the repo | PASS | §5.6 |

### Outputs

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| O1 | Runnable app | PASS | §5.1, §5.2 |
| O2 | README (setup, run, architecture, model choices, limitations) | PASS | `README.md` |
| O3 | Architecture doc / diagram | PASS | `docs/architecture.md` (Mermaid). The diagram order was wrong (fixed, §8) |
| O4 | Evaluation template | PASS | `docs/evaluation_template.md` (all required columns) |
| O5 | Demo walkthrough | PASS | `docs/demo_walkthrough.md` |
| O6 | Remaining-steps list | PASS | `docs/submission_checklist.md` |

### Evaluation

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| V1 | 10–15 questions on the actual PDFs | PASS | 14 in `evaluation/questions.json` |
| V2 | Categories (factual, multi-chunk, synthesis, not-present, failure-revealing) | PASS | F1–4, M1–3, S1–2, U1–3, A1–2 (adversarial = failure-revealing). There is no dedicated *partially answerable* question (U3 comes closest) |
| V3 | Expected answers verified against the PDFs with pages | PASS | All 14 checked by the auditor (§4). M1 had a document inconsistency that was not noted (fixed in notes, §8) |
| V4 | Per-case fields | PASS | template columns `docs/evaluation_template.md:74` cover every required field |
| V5 | Failure classes R / C / G / A | PASS | `docs/evaluation_template.md:104-115` |
| V6 | No fabricated results; human fields not auto-filled | PASS | both JSONs: every `human_judgement` value is `""` (checked by script); 28 blank judgement rows in each `.md`; template all `TBD`; README claims no results |
| V7 | Evaluation actually completed and reported | **PARTIAL (student task)** | Two automatic runs exist (top-k 4 and 8). Human judgement and the summary metrics are not filled in |

### Acceptance

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| X1 | App starts as described in the README | PASS | §5.1 |
| X2 | Tests pass | PASS | 98/98 |
| X3 | No unobserved results reported | PASS | V6 |

**Totals:** 62 items: **57 PASS, 5 PARTIAL (A9, E3, E5, E6, V7), 0 FAIL.** The caveats on A7, D5 and F8 are
noted but do not change those items from PASS.

## 4. Ground-truth spot-check (all 14 questions)

Method: `pypdf` text of every page, whitespace collapsed, regex for the key quoted phrase, and
the 1-based page reported. Script: `/private/tmp/claude-501/audit/gt.py` (outside the repo).

| ID | Phrase checked | Pages found | Stated pages | OK? |
|---|---|---|---|---|
| F1 | "stack of N = 6 identical layers"; "dmodel = 512" | 3; 3, 5 | 3 | Yes |
| F2 | "Total Param-eters=110M ... Total Parameters=340M"; "110M parameters ... 340M parameters" | 3; 8 | 3, 8 | Yes |
| F3 | "December 2018 dump"; "21M documents" | 4; 4 | 4 | Yes |
| F4 | "65 hours"; "5 sec-onds with SBERT" | 1, 2, 8; 1 (hyphenated), 2, 8 | 1, 2 | Yes (also on p.8) |
| M1 | "28.4 BLEU"; "41.8"; "8 P100 GPUs"; "300,000 steps" | 1; 1, 8; 8; 7 | 1, 7, 8 | Yes, **but** p.8 body text says "our big model achieves a BLEU score of 41.0" while the abstract and Table 2 say 41.8. The paper contradicts itself, which was not noted (fixed in notes, §8) |
| M2 | "15% of the token positions"; "80% of the time"; "IsNext" | 4; 4, 12; 4, 13 | 4, 12 | Yes |
| M3 | "same retrieved document to generate the complete sequence"; "different latent document for each target token" | 3; 3 | 3, 4 | Yes |
| S1 | "worse than averaging GloVe"; "default conﬁg-uration is MEAN"; "54.81"; "29.19" | 1; 3; 4; 4 | 1, 3, 4 | Yes |
| S2 | "BERTBASE document encoder"; "BART-large"; "400M parameters"; "110M parameters each"; "keep the document encoder (and index) ﬁxed" | 3; 3,17,18; 3; 18; 4 | 3, 4, 18 (+BERT p.3) | Yes |
| A1 | "dispensing with recurrence and convolutions"; "eschewing recurrence" | 1; 2 | 1, 2 | Yes |
| A2 | "The default conﬁg-uration is MEAN" | 3 (also p.4 "default pooling strategy is MEAN") | 3 | Yes |
| U1 | `GPT-?4`, `GPT-?3` | none in any of the 4 PDFs | — | Absent, confirmed |
| U2 | `MMLU` | none | — | Absent, confirmed |
| U3 | `CO2\|carbon\|emission\|kWh\|tonnes\|energy consumption\|electricity` | only "Carbonell" (author name, 1908.10084 p.11) | — | Absent, confirmed |

Note: the phrases "default configuration is MEAN" and "document encoder ... fixed" only match once
the `ﬁ` ligature is accounted for. This is the same ligature issue as A9.

## 5. Live checks (verbatim excerpts)

### 5.1 Streamlit headless start

```
$ .venv/bin/streamlit run app.py --server.headless true --server.port 8598   (background)
$ curl -s localhost:8598/_stcore/health
ok
$ curl -s -o /dev/null -w "%{http_code}" localhost:8598/
200
log: "Uvicorn server started on :::8598 ... You can now view your Streamlit app in your browser."
```
No errors in the log. The process was stopped afterwards (`pkill`). The health endpoint does not
execute `app.py`, so the script was also driven with real components (§5.2).

### 5.2 Real app run via `streamlit.testing.v1.AppTest` (real embeddings, live Ollama, temp Chroma dir)

```
exception: ElementList()            # i.e. none
sidebar success: ['Ollama ready: llama3.2:3b - Ollama is running and model `llama3.2:3b` is available.']
checkboxes: ['arxiv_1706.03762.pdf (52 chunks)', 'I understand this deletes the whole index']
Q: How many identical layers are in the Transformer encoder?
ANSWER: The Transformer encoder is composed of a stack of N = 6 identical layers [1].
sources: [{'source': 'arxiv_1706.03762.pdf', 'page': 3}, {'source': 'arxiv_1706.03762.pdf', 'page': 5}]
captions: ['**Sources:** arxiv_1706.03762.pdf (p. 3); arxiv_1706.03762.pdf (p. 5)']
expanders: ['Retrieved evidence (4 chunks, top-k=4)']
```
(p.5 is listed although only [1] = p.3 was cited; see E6.) AppTest cannot drive
`st.file_uploader`, so the upload widget itself was not exercised. Ingestion was exercised
through `ingest_file`, the same function the button calls.

### 5.3 Ingestion error handling (`ingest_file`, real embeddings)

```
INGEST arxiv_1706.03762.pdf indexed 15 52 | Indexed 52 chunks from 15 pages.
INGEST arxiv_1810.04805.pdf indexed 16 83 | Indexed 83 chunks from 16 pages.
INGEST arxiv_1908.10084.pdf indexed 11 56 | Indexed 56 chunks from 11 pages.
INGEST arxiv_2005.11401.pdf indexed 19 93 | Indexed 93 chunks from 19 pages.
DUP duplicate | This file is already indexed; skipped re-indexing.          (same bytes as "copy_of_attention.pdf")
TXT-as-PDF error | 'notes.pdf' does not look like a real PDF (missing %PDF- header). It may be another file type renamed to .pdf.
TRUNCATED error | 'broken.pdf' could not be read; the file appears to be corrupted or malformed.
GARBAGE-W-HEADER error | 'garbage.pdf' could not be read; the file appears to be corrupted or malformed.
TXT-EXT error | 'notes.txt' is not a PDF file. Only files with a .pdf extension are supported.
EMPTY error | 'empty.pdf' is empty (0 bytes).
NO-TEXT error | No extractable text was found in 'blank_scan.pdf'. It is probably a scanned or image-only PDF; OCR is not supported in this version.
RESET count 0
```
Cosmetic: pypdf prints `EOF marker not found` to stderr for the truncated file.

### 5.4 Live questions through `RAGPipeline` (top-k 4, all 4 PDFs)

```
### Q1 factual: What dropout rate was used for the base Transformer model?
ANSWER: The dropout rate used for the base Transformer model is 0.1 [3].
ABSTAINED: False SOURCES: [1706.03762 p8, 1706.03762 p9]          -> correct, [3] (p.9 Table 3) contains 0.1

### Partial: How many identical layers does the Transformer encoder have, and what was the rental
### price per GPU-hour used to compute the training cost?
ANSWER: The Transformer encoder has 6 identical layers [1].
I could not find enough information in the uploaded documents to answer the second part of your question regarding the rental price per GPU-hour used to compute the training cost.
ABSTAINED: False                                                   -> correct partial behaviour

### Partial: What percentage of tokens does BERT mask during pre-training, and what was the electricity bill?
ANSWER: The percentage of tokens that BERT masks during pre-training is 80% [2], which means that 20% of the time, the target token remains unchanged [3]. The context does not mention the electricity bill for pre-training BERT.
                                                                   -> states the unsupported part (good) but the fact is WRONG (should be 15%; 80% is the [MASK]-replacement share) = Generation error

### Partial: What optimizer was used ... and how much did the training cost in US dollars?
ANSWER: I could not find enough information in the uploaded documents to answer this question.
### Which optimizer was used to train the Transformer?
ANSWER: I could not find enough information in the uploaded documents to answer this question.
                                                                   -> Retrieval miss: the Adam passage (1706.03762 p.7, "5.3 Optimizer") is not in the top-4; false abstention

### Unanswerable: Who won the 2022 FIFA World Cup?
ANSWER: I could not find enough information in the uploaded documents to answer this question.  ABSTAINED: True SOURCES: []

### Corpus restriction: BERT and RAG papers deselected; "How many total parameters does BERT-Large have?"
ANSWER: I could not find enough information in the uploaded documents to answer this question.
retrieved: only arxiv_1908.10084.pdf chunks                         -> active-corpus filter works; no general-knowledge leak
```

### 5.5 Init-error messages

```
LLMInitError: LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set. Add `OPENAI_API_KEY=sk-...` ...
check_ollama(base_url=http://localhost:1) -> (False, 'Ollama is not running at http://localhost:1 — start it with `ollama serve`.')
check_ollama(model=nosuchmodel:1b)       -> (False, 'Model `nosuchmodel:1b` is not pulled — run `ollama pull nosuchmodel:1b`.')
```

### 5.6 Secrets and README commands

- `grep -rnIE "sk-[A-Za-z0-9]{10,}|api[_-]?key\s*=\s*'...'|hf_...|AKIA...|password\s*="` (excluding `.venv`,
  `.git`, `chroma_db`): the only hit is the test fixture `writer.encrypt(user_password="secret", ...)`.
  It is harmless. `git log -p --all`: no keys. No `.env` file exists. `git ls-files -ci --exclude-standard`: nothing
  ignored is tracked.
- `pytest` / `pytest -m "not slow"`: work as written (97 passed, 1 deselected).
- `python -m evaluation.run_eval --ids F1,U1` (run with `--out` pointing to a scratch dir so the repo
  was not touched): works; F1 answered with the expected page retrieved, U1 abstained; wrote JSON + MD.
- `python -m evaluation.run_eval --help`: works.

## 6. Evaluation draft assessment (auditor's opinion — student must verify)

Runs reviewed: `evaluation/results/results_20261011_113012.*` (top-k 4) and
`results_20261011_113211.*` (top-k 8). Both used llama3.2:3b, temp 0, MiniLM, chunks 1000/200, and
collection `eval_cs1000_co200` (284 chunks). Mean latency was 7.13 s (k=4) and 12.20 s (k=8), with
0 errors. **Human-judgement fields in both files are blank (verified), as they should be.**

"Evidence in context" means the auditor read the retrieved chunk text for the quoted ground-truth
sentence. It is not the automatic `expected_page_retrieved` hint, which is wrong for **A2 at k=8**
(the hint says Y, but the p.3 chunk retrieved is the Figure 2 / objective-function chunk, not the
"default configuration is MEAN" sentence).

| ID | k | Evidence in context? | Correct? | Relevant? | Supported? | Correct abstention? | Failure class | Auditor observations |
|---|---|---|---|---|---|---|---|---|
| F1 | 4 | Y (r1, r2 p.3) | Y | Y | Y | N/A | None | 6 layers [1], d_model 512 [2]. Sources also list p.5 (not cited) |
| F1 | 8 | Y | Y | Y | Y | N/A | None | Same answer |
| F2 | 4 | Y (r1 p.8, r3 p.3) | Y | Y | Y | N/A | None | 110M / 340M [3]. Sources also list RAG p.18 (not cited) |
| F2 | 8 | Y | Y | Y | Y | N/A | None | Same |
| F3 | 4 | **N** (p.4 never retrieved) | N | Y | N | N/A | **Retrieval** (+Generation) | "21M dimensional vectors" misreads "21M 728 dimensional vectors" (p.19); no Wikipedia/Dec-2018 source given; unsupported guess "at least 5 documents" |
| F3 | 8 | **N** | N | Y | Partial | N/A | **Retrieval** | Same misreading; says the document count "is not specified" (honest given its reading, but p.19 implies 21M) |
| F4 | 4 | Y (r1, r2, r3, r4) | Y | Y | Y | N/A | None | 65 h [1] vs 5 s [3] |
| F4 | 8 | Y | Y | Y | Y (citation off) | N/A | None | Cites [5] for "65 hours", but r5 does not contain it (r1, r2, r4, r8 do). Minor citation error |
| M1 | 4 | Y (all p.8) | Y* | Y | Y | N/A | None | 28.4 / **41.0** / 3.5 days on 8 P100. *41.0 is what the p.8 body text says (Table 2 and the abstract say 41.8). Accept, and record the inconsistency |
| M1 | 8 | Y (+p.1 abstract r6) | Y* | Y | Y | N/A | None | Same; BLEU sentence has no citation |
| M2 | 4 | **N** for the details (15%/80-10-10/IsNext chunk not in top-4) | Partial | Y | Partial | N/A | **Retrieval** | Names MLM and NSP but gives no percentages; NSP description mixes in segment embeddings |
| M2 | 8 | Partial (masking chunk r5 yes; NSP 50/50 chunk no) | Partial | Y | Y (stated parts) | N/A | **Retrieval** (+Generation) | 80/10/10 correct [5]; leaves out 15% (it is in r5) and NSP 50/50; adds the full fallback sentence at the end even though it answered (`abstained=False`, correct flag) |
| M3 | 4 | Y (r3 p.3) | Y | Y | Y | N/A | None | Correct contrast; citation only at the end |
| M3 | 8 | Y | Y | Y | Y | N/A | None | Same |
| S1 | 4 | Partial (p.1 "worse than GloVe", p.7 cosine explanation + siamese mention; no MEAN / 54.81) | Partial (weak) | Y | Partial | N/A | **Generation** | Garbles p.7: says raw BERT outputs "treat all dimensions equally" (the paper says *cosine* does). Never explains how SBERT fixes it and does not say that part is missing |
| S1 | 8 | Y (r5 siamese, r7 54.81/29.19, r4 GloVe) | Partial (weak) | Y | Partial | N/A | **Generation** | Same answer despite more evidence. A larger top-k did not help |
| S2 | 4 | Y (r2 p.18 BERT-base enc. + "BART-large"; r4 p.3 BERTBASE) | Partial | Y | Y (stated parts) | N/A | **Generation** | BERT-base query/doc encoders right; generator only described as "pθ", BART never named although it is in r2. BERT-paper cross-reference not retrieved (not essential) |
| S2 | 8 | Y (+r5 p.4 "BART generator") | Partial | Y | Partial | N/A | **Generation** | Adds 110M each [2]; still no BART; "based on the retriever" is wrong |
| U1 | 4/8 | N/A | Y | Y | Y | **Y** | None | Fallback returned word for word |
| U2 | 4/8 | N/A | Y | Y | Y | **Y** | None | Fallback returned word for word |
| U3 | 4/8 | N/A | Y | Y | Y | **Y** | None | Fallback returned word for word, despite related "3.5 days on 8 P100" context |
| A1 | 4 | Partial (p.6 r4 compares self-attention with recurrent layers; p.1/p.2 quote not retrieved) | Y (abstained) | Y | Y | **Y** (abstain is acceptable per rubric) | None | Overly cautious: it could have corrected the premise. A stricter grader might call it a false abstention |
| A1 | 8 | Y (r6 p.10 "replacing the recurrent layers ... with multi-headed self-attention") | Y (abstained) | Y | Y | **Y** (rubric) | None (or Abstention, strict) | Correcting evidence was present but the model abstained |
| A2 | 4 | **N** (all 4 chunks from the BERT paper) | Y (effective abstention) | Y | Y | **Y** (rubric) | None for the outcome (Retrieval noted) | Answer "The passage does not explicitly state why ... I could not find enough information ..." is an abstention in substance, but `abstained=False` and 3 sources are shown. Pipeline detection issue (E6) |
| A2 | 8 | **N** (hint says Y, but the MEAN sentence is not in any chunk) | **N** | Y | **N** | **N** | **Retrieval** (+Abstention) | Accepts the false premise and invents a rationale from BERT `[CLS]` passages; misattributes BERT facts to SBERT; adds a spurious "Note: The question ... is repeated in the context." |

**Auditor's draft summary (verify before using):**

| Metric | top-k 4 | top-k 8 |
|---|---|---|
| Evidence fully in context (F/M/S, 9) | 6/9 (F3 N, M2 N, S1 partial) | 7/9 (F3 N, M2 partial) |
| Correct = Y (F/M/S, 9) | 5/9 (F1, F2, F4, M1, M3); Partial 3 (M2, S1, S2); N 1 (F3) | 5/9; Partial 3; N 1 (same items) |
| Supported = Y (answered, non-abstained F/M/S, 9) | 6/9 | 6/9 |
| Correct abstention (U, 3) | 3/3 | 3/3 |
| Adversarial handled (A, 2) | 2/2 (both abstained, none corrected the premise) | 1/2 (A2 failed) |
| False abstentions on F/M/S | 0/9 | 0/9 |
| Failure classes (earliest stage, non-None) | R: F3, M2 · G: S1, S2 | R: F3, M2, A2 · G: S1, S2 |

Main takeaways for the student's discussion. Retrieval misses (F3, M2, A2) come from dense-only
search. Short "spec" sentences like "21M documents" or "default configuration is MEAN" lose out to
longer, topically similar chunks, and A2 matched the *BERT* paper for "Sentence-BERT ... [CLS]".
Generation misses (S1, S2) are a 3B model leaving out requested parts. Going from top-k 4 to 8
helped M2 (masking details) but caused A2 to fail by adding distracting `[CLS]` passages, and it
added about 5 s per question.

## 7. Issues found (severity-ranked)

| Sev. | Issue | Where | Recommendation |
|---|---|---|---|
| High (submission) | Manual evaluation not completed; template all `TBD` | `docs/evaluation_template.md` | Fill it from the two results files (use §6 as a starting point, but judge every row yourself) |
| Medium | Sources list = all retrieved pages, not the cited passages | `rag/pipeline.py:51-60,107` | Parse `[n]` markers from the answer and show only those chunks' (file, page); fall back to all retrieved if there are no markers |
| Medium | Abstention detected only at the start of the answer; a hedged fallback (A2 at k=4) is shown with sources | `rag/pipeline.py:39-48` | Also treat as abstention when the fallback sentence appears and the answer has no `[n]` citations |
| Medium | Ligatures (`ﬁ`, `ﬂ`, `ﬀ`, `ﬃ`, `ﬄ`) not normalized, so the tokenizer splits them (`ﬁ ##xed`) and embeddings get worse | `rag/pdf_loader.py:78-87` | Add `text = text.translate({0xFB00:"ff",0xFB01:"fi",0xFB02:"fl",0xFB03:"ffi",0xFB04:"ffl"})` (or `unicodedata.normalize("NFKC", ...)`) and a test; then **re-run the evaluation with `--reindex`** |
| Medium | Hyphen re-join also merges real compounds split at a line break ("Englishto-German") | `rag/pdf_loader.py:72,83` | Accept as a known limitation and document it, or only join when the lowercase continuation is not itself a word |
| Medium | Dense retrieval misses short factual sentences (F3, M2, the "optimizer" live question) | design | Hybrid BM25 + dense search or a re-ranker (already listed as future work) |
| Low | Duplicate check runs after the full PDF parse (wasted work; diagram showed the opposite) | `rag/ingest.py:49-54` | Check `has_file(compute_file_hash(data))` before parsing. Diagram fixed (§8) |
| Low | LLM runtime errors other than connection errors (e.g. Ollama "model not found") reach the UI as "Unexpected error: ResponseError ..." | `rag/generation.py:161-169` | Map `ResponseError` to `LLMInitError` with a pull hint |
| Low | pypdf prints `EOF marker not found` to stderr for corrupted files | `rag/pdf_loader.py:93` | Cosmetic; silence the `pypdf` logger |
| Low | `expected_page_retrieved` hint can be Y when the wrong chunk from the right page was retrieved (A2 at k=8) | `evaluation/run_eval.py:73-82` | Already labelled a hint; the student must judge evidence from chunk text |
| Low | Dependencies are unpinned (`>=`) | `requirements.txt` | Add `pip freeze > requirements-lock.txt` for reproducibility |
| Low | No dedicated partially-answerable question in the eval set | `evaluation/questions.json` | Optional: add one (e.g. the "layers + GPU-hour price" question from §5.4) |

## 8. Fixes applied by the auditor

The test suite was re-run after these changes: **98 passed**. No code behaviour was changed. Nothing was committed.

1. **`docs/architecture.md`, ingestion diagram.** The flowchart showed the duplicate check *before*
   PDF parsing, and encryption as part of `validate_pdf_bytes`. In the code, `validate_pdf_bytes`
   checks extension, header, emptiness and size; then pypdf checks for corruption and encryption;
   then text is extracted; and only then does `store.has_file` run (`rag/ingest.py:28-54`). The
   diagram now matches the code.
2. **`.gitignore`.** Removed `evaluation/results/*.json`. The checklist says to "Keep the produced
   `results_<timestamp>.json`", and the JSON files are the machine-readable evidence of the runs,
   but they were being excluded from git. They now show as untracked files ready to commit.
3. **`evaluation/questions.json`, M1.** Added a note to `expected_answer` and `notes`: the
   Transformer paper states EN-FR BLEU as **41.8** (abstract p.1, Table 2 p.8) and also as **41.0**
   (body text p.8). Both runs answered 41.0 from the p.8 text, so the ground truth should not mark
   that wrong. The historical results files still hold the old wording, which is expected.
   **Please also update the M1 row in `docs/evaluation_template.md` yourself** (the auditor was told
   not to edit that file).
4. **`README.md`.** Added `docs/audit_report.md` to the project-structure tree.

Deliberately **not** fixed (they change system behaviour after the evaluation runs were recorded,
so the student should decide and then re-run the evaluation): the citation-precise sources
(`pipeline.py`), abstention detection, ligature normalization, and the dedupe ordering.

## 9. Remaining gaps and risks for submission

1. **Complete the human evaluation** from the existing runs, or from new runs if you apply any of
   the medium fixes. Re-run with `--reindex` after changing normalization, because the eval
   collection name does not change when the code changes. Report counts with denominators.
2. **Sources over-inclusion** could be questioned by an examiner ("Sources: … p.18" for a page the
   answer did not use). Either fix it (Section 7) or explain it in the report and demo: the
   `[n]` numbers map to the evidence panel's `#n`.
3. **A2 at top-k 8 is a clear hallucination case.** Present it honestly as a Retrieval + Abstention
   failure. It is good material for the error analysis.
4. **The demo script says to expect premise correction for A2.** At top-k 4 the system abstains
   (with sources shown). Rehearse so the narration matches what actually appears on screen.
5. **Reproducibility.** Dependencies are unpinned, and Ollama model output can drift between
   versions. Record `pip freeze` and `ollama --version` in the report.
6. **Upload widget not automatically tested** (AppTest limitation). Do one manual upload in the
   browser (4 PDFs + `notes.pdf` + `broken.pdf`) before recording the demo.
7. Before zipping, delete `chroma_db/` (gitignored) so examiners start from a clean index, as the
   checklist says.

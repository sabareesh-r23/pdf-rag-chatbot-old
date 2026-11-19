# Submission Checklist

Status as of 2026-11-20 (Version 1.3). `[x]` = done and verified in this repository; `[ ]` = still to do.

## Still to do (student)
- [ ] Optional: record a voice-over for `submission/RAG_PDF_Chatbot_Demo.mp4` using
      `docs/demo_voiceover.md` (timed to the current video), then replace the file and push.
- [ ] Optional: a final look at the deck in PowerPoint (fonts may differ slightly from the LibreOffice render).
- [ ] Submit the GitHub link (plus the deck, video and PDFs if the portal asks for files).

## 1. Code
- [x] All core requirements implemented (see `docs/requirements_matrix.md`).
- [x] `.venv/bin/pytest -q`: **139 passed**, including the slow real-embedding test (similarity and hybrid).
- [x] Final default: hybrid BM25 + embedding search, cross-encoder reranking (30 → 4), running headers
      stripped, `llama3.2:3b` at temperature 0. Similarity, MMR, query decomposition and all LLM
      parameters (temperature, top-p, sampling top-k, context window) are adjustable in the sidebar.
- [x] Independent review (2026-11-20) found two real bugs, both fixed with regression tests: MMR did
      nothing while reranking was on; a stray one-letter BM25 token could win the keyword ranking.
- [x] Browser pass in real Chrome (Playwright): upload widget, index, questions, evidence panel, untick a
      file, settings, duplicate upload, re-index, clear corpus. No errors.
- [x] Clean clone from GitHub + install from `requirements.txt` + tests: 139 passed, defaults and reranker load verified (2026-11-20).
- [x] No secrets in the repo (`.env` git-ignored).

## 2. Evaluation
- [x] 33 questions with ground truth verified against the PDFs: corpus A (15), B (9), C (9, different domain).
- [x] V1.1 baseline, V1.2 MMR, V1.2 corpus C, V1.3 ablation (header stripping, hybrid, hybrid + rerank),
      top-k 6, LLM comparison (`qwen2.5:3b`, `llama3.1:8b`), query decomposition, and the final runs of
      the submitted code (`results_20261120_010040` / `010255` / `010426`), all judged in the report.
- [x] Headline: wrong answers 7 → 2 of 22 (correct 11 → 12); all 7 unanswerable questions declined.
- [x] Retrieval-only experiment committed and reproducible: `python -m evaluation.retrieval_experiment`.
- [x] Ideas tried and not adopted are documented with evidence (table extraction, query decomposition,
      premise check; report §8e Step 4).

## 3. Presentation
- [x] Deck: `docs/presentation/RAG_PDF_Chatbot_Capstone.pptx` (16 slides, speaker notes; numbers from
      the evaluation report). Source: `docs/presentation/source/` (`deck_data.json` + `build_deck.js`).
- [x] Every slide rendered (LibreOffice) and checked; text fits; validator passes.
- [x] Slide 8 shows a screenshot of the final app (rerank scores in the evidence panel).

## 4. Demo video
- [x] `submission/RAG_PDF_Chatbot_Demo.mp4` (5:20, 1440×900, captioned screen recording of the final
      app): upload and index 4 PDFs; factual, multi-part, partially answerable, unanswerable and
      false-premise questions; evidence panel with rerank scores; active-corpus filter; settings; a
      two-passage answer; the NIST document (D1, which failed before the fixes, now answered; fines declined). Every scene checked frame by frame against the answers.
- [x] Voice-over script timed to this video: `docs/demo_voiceover.md`.

## 5. Dataset and packaging
- [x] Corpus A (61 pp.), B (152 pp.), C (32 pp.) in `data/`; `data/download_datasets.sh` re-downloads them.
- [x] Input PDFs are shared, as asked in the class session (2026-11).
- [x] Public repository: https://github.com/sabareesh-r23/pdf-rag-chatbot
- [x] `SUBMISSION.md` lists every deliverable and where it is.

# Submission Checklist

Status as of 2026-11-19 (Version 1.2: MMR retrieval and explicit LLM parameters added after the 2026-11 class session). `[x]` = done and verified in this repository; `[ ]` = still to do.
Items marked **(you)** need the student; items marked **(professor)** need confirmation from the professor.

## 1. Code
- [x] All core requirements implemented (see `docs/requirements_matrix.md`).
- [x] `.venv/bin/pytest -q`: 123 passed (V1.2), including the slow real-embedding test.
- [x] V1.2: MMR search type and temperature / top-p / context-window controls in the sidebar, `.env` and `run_eval.py`.
- [x] App starts headless (`streamlit run app.py`, health endpoint OK; the audit drove a real question through it).
- [x] Browser pass in real Chrome (automated with Playwright): upload with the widget, index, ask F1, open
      evidence, untick a file, switch to MMR and back, duplicate upload, re-index, clear corpus. No errors. The upload widget and re-index are not covered by automated tests.
- [x] No secrets in the repo (`.env` git-ignored; audit checked the files and git history).
- [ ] Optional: `pip freeze > requirements-lock.txt` for exact reproducibility.

## 2. Evaluation
- [x] 15 + 9 questions with ground truth verified against the PDFs (quotes and pages in the JSON files).
- [x] Final runs at top-k 4 and 8 for both corpora (`evaluation/results/results_20261011_1217*`–`1224*`);
      V1 baseline runs kept; intermediate runs in `results/superseded/` with an explanation.
- [x] Different-domain check: 9 questions on NIST CSF 2.0 (corpus C), 3 runs, judged in report §8d.
- [x] V1.2 MMR runs at top-k 4 and 8 for both corpora (`results_20261119_22*`), judged in report §8b;
      LLM parameters and the measured context use in §8c.
- [x] `docs/evaluation_report.md` written from the real outputs, with a rationale for each judgement.
- [x] Judgements re-checked; the three borderline rows (S1 k4, F3 k8 MMR, LM2 k8 MMR) decided by the §1
      rubric with the reasoning in each row; status banner set to final.
- [ ] Optional: LLM comparison (e.g. `--llm-model llama3.1:8b` if RAM allows, or an API model). Not done.
      The top-k 4 vs 8 comparison is a retrieval comparison, not an LLM comparison.

## 3. Presentation
- [x] Deck: `docs/presentation/RAG_PDF_Chatbot_Capstone.pptx` (15 slides, speaker notes; numbers
      taken from the evaluation report). Source: `docs/presentation/source/`.
- [x] Every slide rendered (LibreOffice) and reviewed; text fits.
- [ ] Optional: a final look in PowerPoint itself (fonts may render slightly differently from LibreOffice).
- [x] Slide 8 shows a real app screenshot (`docs/presentation/source/ui_screenshot.png`).
- [ ] Only if a judgement changes later: update slides 10–12 via `deck_data.json` and rebuild.

## 4. Demo video (5–10 min)
- [x] `submission/RAG_PDF_Chatbot_Demo.mp4` (4:56, 1440×900, captioned screen recording of the real app:
      upload and index 4 PDFs, factual / multi-part / partial / unanswerable / false-premise questions,
      evidence panel, active-corpus filter, settings, MMR). No voice-over; optionally record narration over it.
- [x] Script updated to V1.1 behaviour: `docs/demo_walkthrough.md`.
- [ ] Optional: record your own narrated version (rehearse once; the first answer after `ollama serve` is slower).

## 5. Dataset and packaging
- [x] Corpus A (61 pp.) and corpus B (152 pp.) in `data/`; `data/download_datasets.sh` re-downloads them.
- [ ] **(professor)** Does "50–200 pages" apply per document or per collection?
- [x] Input PDFs are to be shared (class session, 2026-11). Previous question kept for reference: may the PDFs be included in the ZIP? Sentence-BERT is CC BY-SA 4.0; the other
      five use the arXiv non-exclusive distribution licence, which only lets arXiv redistribute. If not
      allowed, ship `download_datasets.sh` instead of the PDFs.
- [ ] Create the archive without generated artefacts:
      `zip -r pdf-rag-chatbot.zip . -x ".venv/*" "chroma_db/*" "*/__pycache__/*" ".pytest_cache/*" ".env" "docs/presentation/source/node_modules/*"`
- [ ] Unzip into a clean folder and follow the README from scratch.

## 5b. Clarified in the class session (2026-11)
- Deliverables: code + `requirements.txt`, PPT (approach and tools, little code), screen-recorded demo
  video (upload a PDF, ask, show answers), the input PDFs used, and a GitHub link. A cloud deployment is
  optional; the video covers it.
- The input PDFs should be shared, which settles the dataset question above for this submission.
- [x] Public repository: https://github.com/sabareesh-r23/pdf-rag-chatbot
- [x] `SUBMISSION.md` lists every deliverable and where it is.

## 6. Final submission
- [x] Repository link: https://github.com/sabareesh-r23/pdf-rag-chatbot
- [x] Evaluation report (final)
- [x] Presentation slides
- [x] Demo video (`submission/RAG_PDF_Chatbot_Demo.mp4`)
- [x] Dataset (`data/`, plus `data/download_datasets.sh`)
- [ ] **(you)** Upload/send these to the course portal before the deadline.

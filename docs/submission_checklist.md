# Submission Checklist

Status as of 2026-11-19 (Version 1.2: MMR retrieval and explicit LLM parameters added after the 2026-11 class session). `[x]` = done and verified in this repository; `[ ]` = still to do.
Items marked **(you)** need the student; items marked **(professor)** need confirmation from the professor.

## 1. Code
- [x] All core requirements implemented (see `docs/requirements_matrix.md`).
- [x] `.venv/bin/pytest -q`: 123 passed (V1.2), including the slow real-embedding test.
- [x] V1.2: MMR search type and temperature / top-p / context-window controls in the sidebar, `.env` and `run_eval.py`.
- [x] App starts headless (`streamlit run app.py`, health endpoint OK; the audit drove a real question through it).
- [ ] **(you)** One manual browser pass: upload PDFs with the widget, ask F1, open evidence, untick a file,
      switch Search type to MMR and back, re-index, clear corpus. The upload widget and re-index are not covered by automated tests.
- [x] No secrets in the repo (`.env` git-ignored; audit checked the files and git history).
- [ ] Optional: `pip freeze > requirements-lock.txt` for exact reproducibility.

## 2. Evaluation
- [x] 15 + 9 questions with ground truth verified against the PDFs (quotes and pages in the JSON files).
- [x] Final runs at top-k 4 and 8 for both corpora (`evaluation/results/results_20261011_1217*`–`1224*`);
      V1 baseline runs kept; intermediate runs in `results/superseded/` with an explanation.
- [x] V1.2 MMR runs at top-k 4 and 8 for both corpora (`results_20261119_22*`), judged in report §8b;
      LLM parameters and the measured context use in §8c.
- [x] `docs/evaluation_report.md` written from the real outputs, with a rationale for each judgement.
- [ ] **(you)** Confirm or correct every draft judgement (especially the rows marked ⚑), then change the
      status banner from DRAFT to confirmed. This includes the new §8b MMR rows (F3 k8 and LM2 k8 are ⚑).
- [ ] Optional: LLM comparison (e.g. `--llm-model llama3.1:8b` if RAM allows, or an API model). Not done.
      The top-k 4 vs 8 comparison is a retrieval comparison, not an LLM comparison.

## 3. Presentation
- [x] Draft deck: `docs/presentation/RAG_PDF_Chatbot_Capstone.pptx` (15 slides, speaker notes; numbers
      taken from the evaluation report). Source: `docs/presentation/source/`.
- [ ] **(you)** Open it in PowerPoint and review every slide visually. It was validated and text-fit
      estimated, but not rendered slide by slide.
- [ ] **(you)** Replace the dashed placeholder on slide 8 with a real app screenshot.
- [ ] **(you)** If you change any judgement in the report, update the result slides 10–12 (12 = MMR vs
      similarity and LLM settings), via `deck_data.json` and a rebuild.

## 4. Demo video (5–10 min)
- [x] Script updated to V1.1 behaviour: `docs/demo_walkthrough.md`.
- [ ] **(you)** Rehearse once (the first answer after `ollama serve` is slower), then record and export MP4.

## 5. Dataset and packaging
- [x] Corpus A (61 pp.) and corpus B (152 pp.) in `data/`; `data/download_datasets.sh` re-downloads them.
- [ ] **(professor)** Does "50–200 pages" apply per document or per collection?
- [ ] **(professor)** May the PDFs be included in the ZIP? Sentence-BERT is CC BY-SA 4.0; the other
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
- [ ] **(you)** Push the repository to GitHub (there is no remote yet) and add the link to the submission.
- [ ] Optional: a one-page index (doc or README section) listing every submitted file and its purpose.

## 6. Final submission
- [ ] Code archive / repository link
- [ ] Confirmed evaluation report
- [ ] Presentation slides
- [ ] Demo video
- [ ] Dataset (or download script, per the professor's answer)

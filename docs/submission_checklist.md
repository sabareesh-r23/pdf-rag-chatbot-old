# Submission Checklist

Remaining steps for the academic submission. Tick each item only when it is actually done.

## 1. Verify the code
- [ ] `pip install -r requirements.txt` in a fresh Python 3.12 venv succeeds.
- [ ] `pytest -m "not slow"` passes; `pytest` (incl. real-embedding tests) passes.
- [ ] `streamlit run app.py` starts; upload, ask, sources, evidence, top-k change and clear corpus all work.
- [ ] Ollama path works (`ollama pull llama3.2:3b`); optionally the `LLM_PROVIDER=openai` path.
- [ ] No secrets in the repository (`.env` excluded; only `.env.example` shipped).

## 2. Run the evaluation and complete the report
- [ ] `python -m evaluation.run_eval --questions evaluation/questions.json --reindex`
- [ ] Keep the produced `evaluation/results/results_<timestamp>.json` and `.md`.
- [ ] Fill every `TBD` in `docs/evaluation_template.md` from the actual results (judgement columns,
      failure classes, observations, summary metrics with denominators).
- [ ] Write the discussion: dominant failure classes, concrete examples, proposed improvements.
- [ ] Remove the "NOT YET COMPLETED" banner only once everything is filled from real outputs.
- [ ] README / slides refer to the completed report instead of quoting unobserved numbers.

## 3. Optional: configuration / model comparison
- [ ] Top-k comparison (e.g. 2 / 4 / 8): `python -m evaluation.run_eval --top-k N`
- [ ] Chunk-size comparison (e.g. 500/100, 1000/200, 1500/300) with `--chunk-size/--chunk-overlap --reindex`
- [ ] Model comparison (e.g. `--llm-model llama3.1:8b` if RAM allows, or an OpenAI-compatible model)
- [ ] Fill Section 6 of the evaluation report and discuss the trade-offs (quality vs. latency vs. memory).

## 4. Presentation (suggested outline, ~10–12 slides)
- [ ] 1. Title, problem statement: answer questions over private PDFs without hallucination.
- [ ] 2. Requirements: upload, grounded answers with citations, abstention, local/free operation.
- [ ] 3. Tech stack: Python 3.12, Streamlit, LangChain, PyPDF, RecursiveCharacterTextSplitter,
      Sentence-Transformers all-MiniLM-L6-v2, Chroma, Ollama llama3.2:3b (OpenAI-compatible optional).
- [ ] 4. Architecture diagram (from `docs/architecture.md`): ingestion path and query path.
- [ ] 5. Implementation: module responsibilities, validation & error handling, file-hash dedupe,
      active-corpus filter, grounded prompt and fallback.
- [ ] 6. Model and tool choices with reasons (local/free/8 GB RAM, cosine on normalized embeddings,
      chunk size 1000/200, top-k 4, temperature 0).
- [ ] 7. Evaluation methodology: dataset, 14 questions in 5 categories, verified ground truth, metrics,
      failure classes.
- [ ] 8. Results: from the completed evaluation report only.
- [ ] 9. Error analysis: representative failures and their class.
- [ ] 10. Limitations and future work (OCR, tables, hybrid retrieval, re-ranking, larger model, multi-user).
- [ ] 11. Live demo or video excerpt; 12. Q&A.

## 5. Demo video
- [ ] Follow `docs/demo_walkthrough.md` (5–10 minutes).
- [ ] Show: startup, upload + ingestion status incl. invalid/corrupted file, factual Q with sources
      and evidence, multi-chunk Q, unanswerable Q with fallback, top-k change, clear corpus.
- [ ] Check audio, resolution, and that terminal/browser text is readable; export as MP4.

## 6. Package code and dataset
- [ ] Remove generated artefacts not needed for grading: `chroma_db/`, `__pycache__/`, `.venv/`,
      `.pytest_cache/` (keep the final evaluation results files).
- [ ] Include `data/sample_pdfs/` (4 public arXiv papers) with attribution (arXiv IDs 1706.03762,
      1810.04805, 1908.10084, 2005.11401) — or a download script/instructions if size is a concern.
- [ ] Include `evaluation/questions.json`, results files, and the completed evaluation report.
- [ ] Include `README.md` and `docs/` (architecture, evaluation report, demo walkthrough).
- [ ] Create the archive, e.g. `zip -r pdf-rag-chatbot.zip . -x ".venv/*" "chroma_db/*" "*/__pycache__/*" ".env"`.
- [ ] Unzip into a clean folder and follow the README from scratch to confirm it works.

## 7. Final submission
- [ ] Code archive / repository link
- [ ] Completed evaluation report
- [ ] Presentation slides
- [ ] Demo video
- [ ] Any required forms / declarations per the course instructions

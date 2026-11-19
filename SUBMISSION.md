# Submission: AI-Powered PDF Chatbot Using RAG

**Repository:** https://github.com/sabareesh-r23/pdf-rag-chatbot

| Deliverable | File | What it is |
|---|---|---|
| Working prototype (code) | `app.py`, `rag/`, `evaluation/`, `tests/` | Streamlit chatbot. Run it with `streamlit run app.py` (setup in [README.md](README.md)); 139 automated tests |
| Requirements | [`requirements.txt`](requirements.txt) | Python dependencies (Python 3.12); `.env.example` lists every setting |
| Presentation (PPT) | [`docs/presentation/RAG_PDF_Chatbot_Capstone.pptx`](docs/presentation/RAG_PDF_Chatbot_Capstone.pptx) | 16 slides with speaker notes: problem and objectives, architecture, tools, data handling, chatbot functionality, evaluation, improvements, insights |
| Demo video | [`submission/RAG_PDF_Chatbot_Demo.mp4`](submission/RAG_PDF_Chatbot_Demo.mp4) | 4:47 captioned screen recording of the running app: upload and index PDFs, factual / multi-part / partially answerable / unanswerable / false-premise questions with cited answers, the evidence panel with rerank scores, the document filter and the settings. Voice-over script: [`docs/demo_voiceover.md`](docs/demo_voiceover.md) |
| Input PDFs | [`data/sample_pdfs/`](data/sample_pdfs/) (4 papers, 61 pages), [`data/long_pdfs/`](data/long_pdfs/) (2 papers, 152 pages), [`data/domain_pdfs/`](data/domain_pdfs/) (NIST CSF 2.0, 32 pages) | The documents used for the demo and the evaluation (public arXiv papers; a public-domain US government document) |
| Evaluation (manual) | [`docs/evaluation_report.md`](docs/evaluation_report.md) | 33 questions with verified expected answers over three corpora; per-question judgements, failure analysis, the improvements they led to (with an ablation), an LLM comparison, and ideas tried and rejected |
| Evaluation run records | [`evaluation/results/`](evaluation/results/) | Raw JSON + Markdown for every run (retrieved chunks, rerank scores, answers, sources, latency) |
| Architecture | [`docs/architecture.md`](docs/architecture.md) | Diagram, module responsibilities, design decisions |
| Requirements coverage | [`docs/requirements_matrix.md`](docs/requirements_matrix.md) | Each requirement mapped to the code and evidence |

## Summary

- **Problem.** General chatbots answer from general knowledge and can invent facts; users of a document
  collection need answers that come only from their PDFs and can be checked.
- **Approach.** PyPDF extraction per page → text normalisation and removal of running headers →
  1000/200-character chunks → all-MiniLM-L6-v2 embeddings → Chroma vector DB → **hybrid search** (BM25
  keywords + embeddings, Reciprocal Rank Fusion) → **cross-encoder reranking** of 30 candidates → top 4 →
  grounded prompt with numbered passages → Llama 3.2 3B via Ollama (temperature 0, top-p 0.9, context
  4096) → answer with [n] citations and file/page sources, or a fixed fallback when the PDFs do not
  contain the answer. Similarity-only and MMR search are selectable in the UI.
- **Results (manual evaluation, 22 answerable + 11 unanswerable/false-premise questions).** The first
  evaluation showed retrieval misses were the main failure, with two verified causes. The fixes that
  targeted them cut **wrong answers from 7 to 2 of 22** (correct 11 → 12), and every unanswerable
  question was still declined, including on a different-domain government document. Remaining errors
  are mostly the 3B model misreading retrieved text; a larger model reads better but did not fit this
  8 GB laptop. Table extraction and query decomposition were tried and rejected with evidence.

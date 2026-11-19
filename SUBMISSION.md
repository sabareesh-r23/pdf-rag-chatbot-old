# Submission: AI-Powered PDF Chatbot Using RAG

**Repository:** https://github.com/sabareesh-r23/pdf-rag-chatbot

| Deliverable | File | What it is |
|---|---|---|
| Working prototype (code) | `app.py`, `rag/`, `evaluation/`, `tests/` | Streamlit chatbot. Run it with `streamlit run app.py` (setup in [README.md](README.md)) |
| Requirements | [`requirements.txt`](requirements.txt) | Python dependencies (Python 3.12); `.env.example` lists every setting |
| Presentation (PPT) | [`docs/presentation/RAG_PDF_Chatbot_Capstone.pptx`](docs/presentation/RAG_PDF_Chatbot_Capstone.pptx) | 15 slides with speaker notes: problem and objectives, architecture, tools, data handling, chatbot functionality, evaluation, insights |
| Demo video | [`submission/RAG_PDF_Chatbot_Demo.mp4`](submission/RAG_PDF_Chatbot_Demo.mp4) | 4:56 captioned screen recording of the running app: upload and index PDFs, questions with cited answers, evidence panel, unanswerable and false-premise questions, document filter, settings and MMR. Voice-over script: [`docs/demo_voiceover.md`](docs/demo_voiceover.md) |
| Input PDFs | [`data/sample_pdfs/`](data/sample_pdfs/) (4 papers, 61 pages), [`data/long_pdfs/`](data/long_pdfs/) (2 papers, 152 pages), [`data/domain_pdfs/`](data/domain_pdfs/) (NIST CSF 2.0, 32 pages) | The documents used for the demo and the evaluation (public arXiv papers; a public-domain US government document) |
| Evaluation (manual) | [`docs/evaluation_report.md`](docs/evaluation_report.md) | 33 questions with verified expected answers over three corpora (two ML-paper sets and a different-domain NIST policy document), run with similarity and MMR; per-question judgements and failure analysis |
| Evaluation run records | [`evaluation/results/`](evaluation/results/) | Raw JSON + Markdown for every run (retrieved chunks, answers, sources, latency) |
| Architecture | [`docs/architecture.md`](docs/architecture.md) | Diagram, module responsibilities, design decisions |
| Requirements coverage | [`docs/requirements_matrix.md`](docs/requirements_matrix.md) | Each requirement mapped to the code and evidence |

## Summary

- **Problem.** General chatbots answer from general knowledge and can invent facts; users of a document
  collection need answers that come only from their PDFs and can be checked.
- **Approach.** PyPDF extraction per page → text normalisation → 1000/200-character chunks →
  all-MiniLM-L6-v2 embeddings → Chroma vector DB → similarity (default) or MMR retrieval, top-k 4 →
  grounded prompt with numbered passages → Llama 3.2 3B via Ollama (temperature 0, top-p 0.9, context
  4096) → answer with [n] citations, file and page sources, or a fixed fallback when the PDFs do not
  contain the answer.
- **Results (manual evaluation).** Every unanswerable question was correctly declined, including on a
  different-domain government policy document. Retrieval misses
  are the main source of wrong answers; MMR helps on long documents but hurts on short papers, so
  similarity is the default. Details and limitations are in the evaluation report.

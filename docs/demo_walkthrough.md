# Demo Video Script (5–10 minutes)

Target length: about 8 minutes. Questions refer to IDs in `evaluation/questions.json`; the verified
expected answers are listed so you can check what the app says live. Do **not** claim results in the
narration that you did not see on screen.

## Before recording (not on camera)

- [ ] `ollama pull llama3.2:3b` done; embedding model already downloaded once (first run downloads ~90 MB).
- [ ] Start from an empty index: run the app once and use **Clear corpus**, or delete `chroma_db/`.
- [ ] Prepare two bad files on the Desktop:
  - `notes.pdf` — a plain text file renamed to `.pdf` (`echo "hello" > notes.pdf`). The uploader only
    accepts `.pdf` extensions, so a renamed file is how to show server-side validation.
  - `broken.pdf` — a truncated real PDF: `head -c 3000 data/sample_pdfs/arxiv_1706.03762.pdf > broken.pdf`.
- [ ] Close unrelated apps (8 GB RAM); set browser zoom so the sidebar and chat are both readable.
- [ ] Do one rehearsal so the model is loaded in memory (first answer after `ollama serve` is slower).

## Script

### 0:00 – 0:45 Introduction
"This is a Retrieval-Augmented Generation chatbot that answers questions only from PDFs you upload,
with file and page citations. It runs fully locally: PyPDF for extraction, LangChain for orchestration,
all-MiniLM-L6-v2 embeddings, a Chroma vector store, and Llama 3.2 3B served by Ollama."
Show `docs/architecture.md` Mermaid diagram briefly (ingestion path vs. query path).

### 0:45 – 1:30 Start the services
```bash
ollama serve            # in terminal 1 (skip if the Ollama app is already running)
ollama list             # shows llama3.2:3b
source .venv/bin/activate
streamlit run app.py    # in terminal 2
```
In the browser, point to sidebar **3. LLM status** showing Ollama is ready with `llama3.2:3b`, and
**2. Active corpus** showing "No documents indexed yet." The chat input is disabled until something is indexed.

### 1:30 – 3:00 Upload and ingestion status (incl. errors)
1. **1. Upload PDFs** -> select all four files from `data/sample_pdfs/` **plus** `notes.pdf` and `broken.pdf`.
2. Click **Index uploaded PDFs**. While it runs, explain: validate -> extract per page -> normalize ->
   chunk (1000/200) -> embed -> store, deduplicated by SHA-256 file hash.
3. In **Last indexing results** point out:
   - the four papers indexed with page and chunk counts;
   - `notes.pdf` rejected because it is not a real PDF (the content is checked, not just the extension);
   - `broken.pdf` rejected as corrupted/unreadable.
4. Upload `arxiv_1706.03762.pdf` again and index -> reported as **duplicate**, not re-indexed.
5. Show **Active corpus**: four files with chunk counts and checkboxes.

### 3:00 – 4:15 Factual question with sources (F1)
Ask **F1**: *"How many identical layers make up the encoder stack of the Transformer in 'Attention Is
All You Need', and what is the model dimension d_model?"*
- Expected (verified): N = 6 layers, d_model = 512 — `arxiv_1706.03762.pdf` p. 3.
- Observed in the final evaluation run (top-k 4): "6 identical layers [1] ... d_model is 512 [2]".
- Point out the inline `[n]` citations and the **Sources** line, which lists only the passages the
  answer cites (here p. 3), not every retrieved chunk.
- Open **Retrieved evidence**: show rank, similarity score, page and the chunk text containing
  "stack of N = 6 identical layers". "Every claim can be checked against the source page."

Do **not** use F3 (RAG's Wikipedia index) as a success example: it is a confirmed retrieval failure.
You can show it deliberately as a failure case: its evidence chunk (p. 4) ranks 22nd for this
wording, so it is not retrieved at top-k 4 or 8.

### 4:15 – 5:00 Multi-passage question (M1)
Ask **M1**: *"What BLEU scores did the big Transformer model achieve on WMT 2014 English-to-German and
English-to-French, and how long and on what hardware was it trained?"*
- Observed: 28.4 (EN-DE), **41.0** (EN-FR), 3.5 days on 8 P100 GPUs.
- Explain: the paper itself is inconsistent. The abstract and Table 2 say 41.8, while the results text
  on p. 8 says 41.0. So 41.0 is supported by the passage that was retrieved. A good example of why
  answers must be checked against the evidence panel.

### 5:00 – 5:30 Partially answerable question (P1)
Ask **P1**: *"How many identical layers are in the Transformer encoder, and how much did it cost in US
dollars to train the big model?"*
- Observed: "6 identical layers [1]" and then a statement that the dollar cost cannot be found.
  This shows the partial-answer rule: answer the supported part, name the unsupported part.

### 5:30 – 6:15 Unanswerable question -> fallback (U1), false premise (A2)
Ask **U1**: *"How many parameters does GPT-4 have?"*
- Expected and observed: the fallback message "I could not find enough information in the uploaded
  documents to answer this question." GPT-4 never appears in these papers; the prompt forbids outside
  knowledge, so abstaining is correct. No sources are shown for an abstention.

Ask **A2** (false premise): *"Why does Sentence-BERT use the [CLS] token output as its default pooling strategy?"*
- The true default is MEAN pooling (1908.10084 p. 3). At top-k 4 the final run abstains (acceptable);
  it does not correct the premise because the "default configuration is MEAN" chunk is not retrieved.
- Be honest if you show top-k 8: in the evaluation the model still produced a fabricated rationale for
  A2 at top-k 8. Present it as a known abstention/hallucination failure (see `docs/evaluation_report.md`).

### 6:15 – 7:15 Change top-k
Open **Settings** -> move **Top-k retrieved chunks** from 4 to 8. Re-ask M2 (BERT pre-training tasks).
- Show that **Retrieved evidence** now lists 8 chunks. In the evaluation, M2's masking details
  (15%, 80/10/10) were only retrieved at top-k 8. Discuss the trade-off: more recall for multi-part
  questions vs. longer prompts, more noise, slower answers, and (for A2) more room for a fabricated answer.
- Mention chunk size/overlap are also configurable and apply on re-index
  (**Re-index with current settings**). Set top-k back to 4.
- Switch **Search type** to **MMR (diverse)** and re-ask LF3 on the long corpus if it is loaded
  ("How accurately could human evaluators identify news articles written by GPT-3 175B?"): in the evaluation, MMR
  reached the 52% table that similarity search missed. Say honestly that MMR hurt on the short papers
  (it drops the neighbouring chunk that continues a passage), so similarity stays the default.
- Point at the **LLM** settings: temperature 0 (faithful, repeatable), top-p 0.9 (no effect at
  temperature 0), context window 4096 tokens (largest prompt measured: about 2,500 tokens).
- Optionally untick a paper in **Active corpus** and ask F2 (BERT parameters) to show that only selected
  files are searched (the answer should fall back if BERT is unticked).

### 7:15 – 7:45 Clear corpus
Tick **"I understand this deletes the whole index"** -> **Clear corpus**. Show the corpus list is empty
and the chat input is disabled again.

### 7:45 – 8:30 Wrap-up
- Evaluation: `python -m evaluation.run_eval` runs the 15-question main set (plus a 9-question set on
  two long papers via `--questions evaluation/questions_long.json --pdf-dir data/long_pdfs`) and writes
  JSON/Markdown records; judged results are in `docs/evaluation_report.md`. Quote only numbers from
  that report once you have confirmed them.
- Limitations: no OCR for scanned PDFs, tables/figures, small local model quality, single-user.
- Possible improvements: hybrid BM25 + dense retrieval, a re-ranker, larger model via `LLM_PROVIDER=openai`.

# Voice-over script for `submission/RAG_PDF_Chatbot_Demo.mp4`

The video is 5:20. Timestamps match the on-screen captions to within a few seconds (checked against
frames taken every 4 s). Each block is sized for a calm pace (about 140 words per minute). If you fall
behind, skip the sentences in *(brackets)*.

**How to record it (Mac):** open iMovie → new project → drag in the MP4 → move the playhead to 0:00 →
Window → Record Voiceover → click the red button and read while the video plays → File → Share → File.
Alternatively, play the video full screen and record the screen with QuickTime (File → New Screen
Recording, Options → Microphone). Save the result over `submission/RAG_PDF_Chatbot_Demo.mp4`.

---

### 0:00 – 0:16 · Title
Hello, this is my capstone project: an AI-powered PDF chatbot built with Retrieval-Augmented Generation.
General chatbots answer from general knowledge and can invent facts. This one answers only from the PDFs
you upload, and cites the file and page for every fact, so every answer can be checked.

### 0:16 – 0:28 · Tools
Everything runs locally on my laptop: Streamlit for the interface, PyPDF to read the PDFs, MiniLM
embeddings and a Chroma vector database, keyword search plus a reranker, and Llama 3.2 3B through
Ollama.

### 0:28 – 0:44 · Upload and index
I upload four research papers, about sixty pages: the Transformer, BERT, Sentence-BERT and RAG papers.
When I click Index, each file is validated, read page by page, cleaned, including removing repeated
page headers, split into chunks of a thousand characters, embedded, and stored with its file name and
page number.

### 0:44 – 0:56 · Indexing report and active corpus
The report shows the pages and chunks per file. Below it is the active corpus: only the files that are
ticked are searched.

### 0:56 – 1:32 · Factual question and evidence
A factual question: how many layers does the Transformer's encoder have, and what is the model
dimension? The system searches in two ways at once, by keywords and by meaning, takes the thirty best
candidates, and a reranker model reads each one together with the question and keeps the best four.
The answer: six layers and a model dimension of 512, with citations, from page 3 of the Transformer
paper.
In Retrieved evidence I can see exactly which passages the model used, with their page, similarity and
rerank scores. *(The top passage contains both facts.)*

### 1:32 – 1:52 · Multi-part question
A multi-part question: the BLEU scores on two translation tasks, and the training time and hardware.
It answers 28.4 and 41.0, and 3.5 days on eight P100 GPUs, citing pages 1 and 8.

### 1:52 – 2:12 · Partially answerable
This question is half answerable: the papers give the number of layers but never a training cost in
dollars. The chatbot answers the first part and says clearly that it cannot find the second, instead
of inventing a number.

### 2:12 – 2:28 · Unanswerable
Something that is not in the documents at all: how many parameters GPT-4 has. It gives its fixed
fallback answer. In my evaluation every unanswerable question was declined like this.

### 2:28 – 2:48 · False premise
This question has a false premise: it asks why the Transformer uses LSTM layers, but it doesn't use
any. The chatbot does not make up a reason; it declines. *(A specific prompt rule handles this.)*

### 2:48 – 3:16 · Active corpus filter
Now I untick the BERT paper. It stays indexed but is no longer searched. I ask a detail that only the
BERT paper contains, how many tokens BERT masks, and since no BERT passage can be retrieved, the
chatbot falls back instead of guessing. Then I tick the paper again.

### 3:16 – 3:36 · Settings
The Settings panel shows the parameters: top-k, the search type, reranking, temperature, top-p, the
context window and chunking. The defaults were chosen by my evaluation: hybrid search with the
reranker, four chunks, and temperature zero for faithful, repeatable answers. *(I measured the largest
prompt at about 2,500 tokens, so the 4,096-token context window never cuts anything off.)*

### 3:36 – 4:12 · Two-passage answer
This answer needs two passages: 65 hours with BERT, from page 1, and about 5 seconds with SBERT, from
page 2. Both are retrieved and cited, and the evidence panel shows their rerank scores.
*(Keyword search helps with exact terms, and the reranker with sentences that are a small part of a
long chunk.)*

### 4:12 – 4:52 · A different kind of document
Now a completely different document: the NIST Cybersecurity Framework, a 32-page government policy
document. I ask for its six Core Functions. In my first evaluation this question failed: every page
repeats the same header, which made all chunks look alike and buried the answer. With the header
removal and the reranker it now lists all six Functions, citing pages 2, 9 and 31.
And when I ask about fines, which the framework does not mention, it declines here too.

### 4:52 – 5:08 · Evaluation
I evaluated it manually with 33 questions on three document sets. Fixing what that evaluation found cut
wrong answers from 7 to 2 out of 22, and every unanswerable question was still declined.

### 5:08 – 5:20 · Close
The report also compares other models and explains what I tried and rejected. Everything is on
GitHub. Thank you for watching.

# Voice-over script for `submission/RAG_PDF_Chatbot_Demo.mp4`

The video is 4:56. Timestamps match the on-screen captions to within a few seconds. Each block is sized
for a calm pace (about 140 words per minute). If you fall behind, skip the sentences in *(brackets)*.

**How to record it (Mac):** open iMovie → new project → drag in the MP4 → move the playhead to 0:00 →
Window → Record Voiceover → click the red button and read while the video plays → File → Share → File.
Alternatively, play the video full screen and record the screen with QuickTime (File → New Screen
Recording, Options → Microphone).

---

### 0:00 – 0:20 · Title
Hello, this is my capstone project: an AI-powered PDF chatbot built with Retrieval-Augmented Generation.
The problem it solves is simple. General chatbots answer from general knowledge and can invent facts.
This chatbot answers only from the PDFs you upload, and it cites the file and page for every fact, so you
can check every answer.

### 0:20 – 0:28 · Tools
Everything runs locally on my laptop: Streamlit for the interface, PyPDF for text extraction, MiniLM
embeddings, a Chroma vector database, and Llama 3.2 3B through Ollama.

### 0:28 – 0:44 · Upload and index
First, I upload four research papers, about sixty pages in total: the Transformer, BERT, Sentence-BERT
and RAG papers. When I click Index, each file is validated, the text is extracted page by page and
cleaned, split into chunks of a thousand characters with two hundred characters of overlap, embedded,
and stored in Chroma together with its file name and page number.

### 0:44 – 0:56 · Indexing report and active corpus
The indexing report shows the pages and chunks for each file, 286 chunks in total. Below it is the
active corpus: only the files that are ticked here are searched when answering.

### 0:56 – 1:32 · Factual question and evidence
Now a factual question: how many layers are in the Transformer's encoder, and what is the model
dimension? The question is embedded with the same model, and the four most similar chunks are
retrieved and passed to the LLM with strict rules: answer only from these passages and cite them.
The answer is six layers and a model dimension of 512, with citations one and two, and the Sources line
points to page 3 of the Transformer paper.
I'll open Retrieved evidence. Here are the exact chunks the model saw, with page numbers and similarity
scores. The top chunk, from page 3, contains both facts. *(This is what makes the answers checkable:
nothing is hidden.)*

### 1:32 – 1:56 · Multi-part question
Next, a multi-part question that needs facts from different parts of the paper: the BLEU scores on two
translation tasks, and how long and on what hardware the model was trained. It answers 28.4 for
English-to-German, 41.0 for English-to-French, and 3.5 days on eight P100 GPUs, all cited to page 8.

### 1:56 – 2:12 · Partially answerable question
This question is only half answerable. The papers give the number of layers but never the training cost
in dollars. The chatbot answers the first part and clearly says it cannot find the second part,
instead of inventing a number.

### 2:12 – 2:28 · Unanswerable question
Now something that is not in the documents at all: how many parameters GPT-4 has. The chatbot gives
its fixed fallback answer: "I could not find enough information in the uploaded documents." In my
evaluation, every unanswerable question was declined like this.

### 2:28 – 2:48 · False premise
This question has a false premise. It asks why Sentence-BERT uses the CLS token by default, but the
paper says the default is mean pooling. The chatbot does not invent a reason; it declines. I added a
specific prompt rule for this after the evaluation showed a failure of this kind.

### 2:48 – 3:16 · Active corpus filter
Next, the document filter. I untick the BERT paper, so it stays indexed but is no longer searched.
I ask a detail that only the BERT paper contains: what percentage of tokens BERT masks. Since no BERT
chunk can be retrieved, the chatbot falls back instead of guessing. Then I tick the paper again.

### 3:16 – 3:48 · Settings and MMR
The Settings panel shows the parameters that control quality. Top-k is the number of chunks
retrieved, four by default. Temperature is zero, which gives faithful and repeatable answers. Top-p is
0.9, and the context window is 4096 tokens. I measured the largest prompt at about 2,500 tokens, so
nothing gets cut off.
The search type can be plain similarity or MMR, Maximal Marginal Relevance. MMR first fetches twenty
candidates, then picks chunks that are relevant but not repetitive. *(I evaluated both.)*

### 3:48 – 4:20 · MMR in action
Here is the same chatbot with MMR. I ask for the parameter counts of BERT-Base and BERT-Large, and it
answers 110 million and 340 million, from page 3 of the BERT paper. In my evaluation, MMR helped on long
documents, like the 75-page GPT-3 paper, because it reaches more sections. It hurt on short papers,
because it drops the neighbouring chunk that continues a passage. So similarity stays the default, and
MMR is an option for long reports.

### 4:20 – 4:36 · Evaluation
I evaluated it manually with 33 questions: two sets of research papers, plus a government cybersecurity
document to check a different domain. Every unanswerable question was correctly declined. Most wrong
answers came from retrieval missing the right passage. *(Hybrid keyword-plus-semantic search and
removing repeated page headers are the next improvements.)*

### 4:36 – 4:56 · Close
The code, the evaluation report and the slides are on GitHub. To sum up: a local chatbot that answers
only from your PDFs, shows its evidence, and says so when the answer isn't there. Thank you for
watching.

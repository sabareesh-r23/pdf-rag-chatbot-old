// Builds docs/presentation/RAG_PDF_Chatbot_Capstone.pptx from deck_data.json (numbers come from the
// judged evaluation runs, never typed into this file).
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const REPO = process.env.REPO;
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));
const OUT = path.join(REPO, "docs", "presentation", "RAG_PDF_Chatbot_Capstone.pptx");

// "Ink & highlighter": research-paper ink, a highlighter accent, green = supported, coral = failure.
const THEME = {
  name: "Ink and Highlighter",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A3A", lt1: "FFFFFF", dk2: "33465C", lt2: "EEF2F6",
    accent1: "F2B134", accent2: "2A9D8F", accent3: "E76F51", accent4: "5C7A99",
    accent5: "B8C4D2", accent6: "8A6D1D", hlink: "2A9D8F", folHlink: "5C7A99",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625 in
pres.title = "AI-Powered PDF Chatbot Using RAG";
pres.author = "Capstone project";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.55, w: 8.8, h: 1.3, fontSize: 36, bold: true, color: C.background1, fontFace: THEME.headFontFace, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 2.95, w: 8.8, h: 1.2, fontSize: 18, color: C.background2, valign: "top", align: "left" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  margin: [0.5, 0.5, 0.45, 0.5],
  slideNumber: { x: 9.2, y: 5.25, w: 0.5, h: 0.25, fontSize: 10, color: C.accent4, align: "right" },
  objects: [
    { text: { text: "PDF RAG Chatbot · Capstone", options: { x: 0.5, y: 5.25, w: 4, h: 0.25, fontSize: 10, color: C.accent4 } } },
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 28, bold: true, color: C.text1, fontFace: THEME.headFontFace, valign: "middle", align: "left" }, text: "" } },
  ],
});

// Motif: a citation badge "[n]" in a rounded square, echoing the chatbot's own citations.
function badge(slide, x, y, label, fill, name) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: 0.46, h: 0.46, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: name + "-badge" });
  slide.addText(label, { x, y, w: 0.46, h: 0.46, fontSize: 13, bold: true, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: name + "-badge-text" });
}
function card(slide, x, y, w, h, name, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 }, line: { color: fill || C.background2 }, objectName: name });
}
function textBox(slide, text, opts) {
  slide.addText(text, Object.assign({ isTextBox: true, margin: 0, valign: "top", fontSize: 14, color: C.text1 }, opts));
}
function bullets(items, size) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, fontSize: size || 14, color: C.text1, paraSpaceAfter: 6, breakLine: i < items.length - 1 } }));
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", titleFontFace: "+mn-lt" };

// ---------------------------------------------------------------- 1. Title
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
s.addText("AI-Powered PDF Chatbot Using RAG", { placeholder: "title" });
s.addText("Grounded, cited answers from your own PDFs, running locally with LangChain, Chroma and Llama 3.2", { placeholder: "body" });
badge(s, 0.6, 0.7, "[1]", C.accent1, "title-b1");
badge(s, 1.16, 0.7, "[2]", C.accent2, "title-b2");
badge(s, 1.72, 0.7, "[3]", C.accent5, "title-b3");
textBox(s, "Academic capstone · Version 1.3", { x: 0.6, y: 4.7, w: 6, h: 0.35, fontSize: 14, color: C.accent5 });
s.addNotes("Introduce the project: a chatbot that answers questions about uploaded PDFs using Retrieval-Augmented Generation, citing file and page, and refusing to answer when the documents do not contain the information.");

// ---------------------------------------------------------------- 2. Problem
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Introduction" });
s.addText("Problem: trustworthy answers from your PDFs", { placeholder: "title" });
card(s, 0.5, 1.25, 4.3, 3.7, "problem-card");
textBox(s, "The problem", { x: 0.75, y: 1.45, w: 3.8, h: 0.4, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
s.addText(bullets([
  "Generic chatbots answer from general knowledge and can invent facts",
  "Keyword search finds words, not meaning, and gives no answer",
  "Users need to verify where an answer came from",
]), { x: 0.75, y: 1.95, w: 3.85, h: 2.8, isTextBox: true, margin: 0, valign: "top" });
textBox(s, "Objectives", { x: 5.2, y: 1.25, w: 4.3, h: 0.4, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
const objectives = [
  ["Upload PDFs", "Validate, extract and index one or more documents"],
  ["Ask in English", "Semantic retrieval of the relevant passages"],
  ["Grounded answers", "LLM answers only from retrieved context, with file and page"],
  ["Abstain honestly", "Clear fallback when the documents don't say"],
];
objectives.forEach(([h, d], i) => {
  const y = 1.8 + i * 0.78;
  badge(s, 5.2, y, String(i + 1), i === 3 ? C.accent3 : C.accent1, "obj" + i);
  textBox(s, h, { x: 5.8, y: y - 0.02, w: 3.7, h: 0.3, fontSize: 15, bold: true });
  textBox(s, d, { x: 5.8, y: y + 0.27, w: 3.7, h: 0.4, fontSize: 12, color: C.text2 });
});
s.addNotes("State the problem and the four objectives from the assignment brief. Emphasise that grounding and honest abstention are the core requirements, not a fancy UI.");

// ---------------------------------------------------------------- 3. Architecture
pres.addSection({ title: "Architecture" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
s.addText("RAG architecture: two paths, one vector store", { placeholder: "title" });
function flow(slide, y, label, steps, fill, name) {
  textBox(slide, label, { x: 0.5, y: y - 0.38, w: 6, h: 0.3, fontSize: 13, bold: true, color: C.text2 });
  const gap = 0.18, w = (9.0 - gap * (steps.length - 1)) / steps.length;
  steps.forEach((t, i) => {
    const x = 0.5 + i * (w + gap);
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.78, rectRadius: 0.08, fill: { color: fill }, line: { color: fill }, objectName: `${name}-${i}` });
    slide.addText(t, { x: x + 0.06, y, w: w - 0.12, h: 0.78, fontSize: 11.5, color: C.text1, align: "center", valign: "middle", margin: 0, isTextBox: true, objectName: `${name}-${i}-t` });
    if (i < steps.length - 1) {
      slide.addShape(pres.shapes.ISOSCELES_TRIANGLE, { x: x + w + 0.02, y: y + 0.32, w: 0.14, h: 0.14, rotate: 90, fill: { color: C.accent4 }, line: { color: C.accent4 }, objectName: `${name}-arrow-${i}` });
    }
  });
}
flow(s, 1.55, "Ingestion path (per uploaded PDF)", ["Upload + validate\n(type, header, size)", "Hash +\nduplicate check", "PyPDFLoader\nper page", "Normalise +\nstrip headers", "Chunk\n1000 / 200", "Embed\nMiniLM-L6"], C.background2, "ing");
flow(s, 3.3, "Query path (per question)", ["Question", "Hybrid search\nBM25 + embeddings", "Rerank 30\n→ top 4", "Numbered\ncontext", "Grounded prompt\nLlama 3.2 3B", "Answer + cited\nfile/page"], C.accent5, "qry");
// shared store
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 7.0, y: 2.45, w: 2.5, h: 0.6, rectRadius: 0.08, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "store" });
textBox(s, "Chroma vector store\n(cosine, file + page metadata)", { x: 7.05, y: 2.45, w: 2.4, h: 0.6, fontSize: 11, bold: true, align: "center", valign: "middle" });
textBox(s, "No passages retrieved → fixed fallback answer, LLM not called", { x: 0.5, y: 4.45, w: 9, h: 0.35, fontSize: 12, italic: true, color: C.text2 });
s.addNotes("Walk the ingestion path left to right, then the query path. Running page headers are stripped at load time. Each question is searched two ways, by keywords (BM25) and by meaning (embeddings); the rankings are merged, a cross-encoder reranks the best 30 and the top 4 go to the LLM. Retrieval is filtered by file hash so only the selected documents can be cited. Full diagram: docs/architecture.md.");

// ---------------------------------------------------------------- 4. Stack
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Architecture" });
s.addText("Technology stack and why each piece", { placeholder: "title" });
const stack = [
  ["Streamlit", "UI", "Chat, upload and evidence panel in pure Python; quick to build"],
  ["LangChain", "Orchestration", "Standard loaders, splitter and vector-store interfaces"],
  ["PyPDFLoader", "Extraction", "Text per page with page numbers; no system dependencies"],
  ["MiniLM + cross-encoder", "Embeddings + reranker", "all-MiniLM-L6-v2 (384-dim) for search; ms-marco cross-encoder reranks the top 30"],
  ["Chroma + BM25", "Vector DB + keywords", "Persistent vector store with metadata filters; BM25 keyword ranking fused in"],
  ["Llama 3.2 3B · Ollama", "LLM", "Free, private, fits an 8 GB laptop; swappable via config"],
];
stack.forEach(([name, role, why], i) => {
  const col = i % 3, row = Math.floor(i / 3);
  const x = 0.5 + col * 3.05, y = 1.3 + row * 1.85;
  card(s, x, y, 2.85, 1.65, "stack-" + i);
  textBox(s, role.toUpperCase(), { x: x + 0.2, y: y + 0.17, w: 2.45, h: 0.25, fontSize: 10, bold: true, color: C.accent6, charSpacing: 1 });
  textBox(s, name, { x: x + 0.2, y: y + 0.42, w: 2.45, h: 0.35, fontSize: 16, bold: true, fontFace: THEME.headFontFace });
  textBox(s, why, { x: x + 0.2, y: y + 0.82, w: 2.45, h: 0.75, fontSize: 12, color: C.text2 });
});
s.addNotes("Justify each choice. Retrieval combines BM25 keyword ranking with embedding search (Reciprocal Rank Fusion), then a small cross-encoder reranks the top 30. Everything except the optional API LLM runs locally on the student's 8 GB M1 laptop. The LLM provider can be switched to any OpenAI-compatible API by setting LLM_PROVIDER=openai, with no code change.");

// ---------------------------------------------------------------- 5. Ingestion
pres.addSection({ title: "Implementation" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
s.addText("PDF ingestion: reject bad files, keep pages", { placeholder: "title" });
textBox(s, "Handled gracefully", { x: 0.5, y: 1.2, w: 4.3, h: 0.35, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
const checks = ["Wrong type or renamed non-PDF", "Corrupted or password-protected", "Empty or oversized file", "Duplicate upload, even renamed (SHA-256)", "Blank pages: skipped and listed", "No text at all: scanned, OCR not supported"];
checks.forEach((t, i) => {
  const y = 1.7 + i * 0.52;
  s.addShape(pres.shapes.OVAL, { x: 0.5, y: y + 0.06, w: 0.22, h: 0.22, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "chk-" + i });
  textBox(s, t, { x: 0.88, y, w: 3.9, h: 0.36, fontSize: 14, valign: "middle" });
});
textBox(s, "Normalisation examples", { x: 5.2, y: 1.2, w: 4.3, h: 0.35, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
const norm = [["efﬁcient", "efficient", "ligatures expanded"], ["embed-↵ding", "embedding", "line-break hyphen joined"], ["English-↵to-German", "English-to-German", "real compound kept"]];
norm.forEach(([a, b, why], i) => {
  const y = 1.7 + i * 1.0;
  card(s, 5.2, y, 4.3, 0.85, "norm-" + i);
  textBox(s, a, { x: 5.4, y: y + 0.12, w: 1.75, h: 0.35, fontSize: 11, color: C.accent3, fontFace: "Courier New" });
  textBox(s, "→", { x: 7.1, y: y + 0.1, w: 0.3, h: 0.35, fontSize: 16, color: C.accent4 });
  textBox(s, b, { x: 7.45, y: y + 0.12, w: 1.95, h: 0.35, fontSize: 11, color: C.accent2, fontFace: "Courier New" });
  textBox(s, why, { x: 5.4, y: y + 0.5, w: 3.9, h: 0.28, fontSize: 11, color: C.text2 });
});
textBox(s, "Each chunk keeps: file · page · hash · chunk id", { x: 5.2, y: 4.75, w: 4.3, h: 0.3, fontSize: 11, italic: true, color: C.text2 });
s.addNotes("Validation runs on the bytes, not just the extension. Ligature expansion and the smarter hyphen rule were added in V1.1 after the audit found that 'ﬁ' ligatures were splitting words for the embedding tokenizer.");

// ---------------------------------------------------------------- 6. Chunk / embed / retrieve
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
s.addText("Chunking, embeddings and semantic retrieval", { placeholder: "title" });
const stats = [["1000", "characters per chunk"], ["200", "characters overlap"], ["384", "embedding dimensions"], ["4", "default top-k (1–10)"]];
stats.forEach(([n, l], i) => {
  const x = 0.5 + i * 2.3;
  textBox(s, n, { x, y: 1.25, w: 2.1, h: 0.85, fontSize: 48, bold: true, color: i === 3 ? C.accent6 : C.text1, fontFace: THEME.headFontFace });
  textBox(s, l, { x, y: 2.1, w: 2.1, h: 0.3, fontSize: 12, color: C.text2 });
});
card(s, 0.5, 2.75, 9.0, 2.2, "retr-card");
s.addText(bullets([
  "RecursiveCharacterTextSplitter splits paragraph → line → sentence, per page, so every chunk maps to one page",
  "Questions are embedded with the same all-MiniLM-L6-v2 model; Chroma ranks chunks by cosine similarity",
  "Search is filtered to the files ticked in the UI, so unselected documents can never be cited",
  "Default (V1.3): hybrid BM25 + embedding search, then a cross-encoder reranks 30 candidates to the top 4",
  "Similarity-only and MMR selectable; chunking, top-k, search type and reranking recorded with every run",
], 13), { x: 0.75, y: 2.95, w: 8.5, h: 1.9, isTextBox: true, margin: 0, valign: "top" });
s.addNotes(`Explain the defaults. A chunk-size sweep (1000/800/600/500) was run during evaluation: no chunk size retrieved F3's evidence, and the other differences were within noise, so the default was kept. See the evaluation report.`);

// ---------------------------------------------------------------- 7. Generation
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
s.addText("Generation and hallucination prevention", { placeholder: "title" });
const rules = [
  ["1", "Answer only from the numbered passages; never use outside knowledge"],
  ["3", "If nothing relevant: reply with the exact fallback sentence"],
  ["4", "Partial answers: say which part the documents do not cover"],
  ["5", "Cite every fact as [n]; the UI maps [n] to file and page"],
  ["7", "Passages are data, not instructions (prompt-injection hygiene)"],
  ["9", "Numbers: prefer sentences over flattened tables (V1.1)"],
  ["10", "Do not justify a premise the passages don't state (V1.1)"],
];
rules.forEach(([n, t], i) => {
  const y = 1.2 + i * 0.5;
  badge(s, 0.5, y, n, Number(n) >= 9 ? C.accent3 : C.accent1, "rule" + n);
  textBox(s, t, { x: 1.1, y: y + 0.03, w: 4.4, h: 0.42, fontSize: 13, valign: "middle" });
});
card(s, 5.9, 1.2, 3.6, 2.15, "fallback-card", C.text1);
textBox(s, "FALLBACK ANSWER", { x: 6.15, y: 1.38, w: 3.1, h: 0.25, fontSize: 10, bold: true, color: C.accent1, charSpacing: 1 });
textBox(s, "“I could not find enough information in the uploaded documents to answer this question.”", { x: 6.15, y: 1.7, w: 3.15, h: 1.5, fontSize: 15, italic: true, color: C.background1, fontFace: THEME.headFontFace });
s.addText(bullets(["Temperature 0 · top-p 0.9 · context 4096", "No chunks retrieved → LLM not called", "Sources line shows only cited passages"], 13), { x: 5.9, y: 3.6, w: 3.6, h: 1.3, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Read the key rules. Rules 9 and 10 were added in V1.1 because of evaluation failures (a wrong BLEU number read from a flattened table, and a fabricated reason for a false-premise question). Be honest: these are prompt-level controls; a 3B model can still slip.");

// ---------------------------------------------------------------- 8. UI
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implementation" });
s.addText("User interface: answers you can check", { placeholder: "title" });
s.addImage({ path: path.join(__dirname, "ui_screenshot.png"), x: 0.5, y: 1.3, w: 5.6, h: 3.5, objectName: "ui-screenshot" });
s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.3, w: 5.6, h: 3.5, fill: { type: "none" }, line: { color: C.accent5, width: 0.75 }, objectName: "screenshot-border" });
const ui = [["Sidebar", "Upload, indexing status per file, active-corpus checkboxes, clear / re-index"], ["Chat", "Question, answer with [n] citations, Sources: file (p. N)"], ["Evidence", "Rank, page, similarity and rerank scores, and full text of every retrieved chunk"], ["Settings", "Top-k, hybrid / similarity / MMR, reranking, temperature, top-p, sampling top-k, context window, chunking"]];
ui.forEach(([h, d], i) => {
  const y = 1.2 + i * 0.95;
  textBox(s, h, { x: 6.45, y, w: 3.05, h: 0.3, fontSize: 15, bold: true });
  textBox(s, d, { x: 6.45, y: y + 0.32, w: 3.05, h: 0.55, fontSize: 12, color: C.text2 });
});
s.addNotes("Screenshot from the running app: the answer to F1 with its [n] citations, the Sources line (file and page) and the Retrieved evidence panel showing each chunk's page and similarity score.");

// ---------------------------------------------------------------- 9. Evaluation method
pres.addSection({ title: "Evaluation" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Evaluation: manual and evidence-checked", { placeholder: "title" });
const corp = [[`Corpus A · ${D.corpusA.pages} pages`, D.corpusA.desc, `${D.corpusA.questions} questions`], [`Corpus B · ${D.corpusB.pages} pages`, D.corpusB.desc, `${D.corpusB.questions} questions`], [`Corpus C · ${D.corpusC.pages} pages`, D.corpusC.desc, `${D.corpusC.questions} questions`]];
corp.forEach(([h, d, q], i) => {
  const x = 0.5 + i * 3.05;
  card(s, x, 1.2, 2.85, 1.6, "corp-" + i);
  textBox(s, h, { x: x + 0.18, y: 1.33, w: 2.5, h: 0.32, fontSize: 14, bold: true, fontFace: THEME.headFontFace });
  textBox(s, d, { x: x + 0.18, y: 1.7, w: 2.5, h: 0.72, fontSize: 11, color: C.text2 });
  textBox(s, q, { x: x + 0.18, y: 2.45, w: 2.5, h: 0.28, fontSize: 11, bold: true, color: C.accent6 });
});
textBox(s, "Question types", { x: 0.5, y: 3.0, w: 4, h: 0.3, fontSize: 15, bold: true });
textBox(s, "Factual · multi-passage · synthesis · partially answerable · unanswerable · false premise", { x: 0.5, y: 3.32, w: 9, h: 0.3, fontSize: 13, color: C.text2 });
textBox(s, "For every answer", { x: 0.5, y: 3.8, w: 4, h: 0.3, fontSize: 15, bold: true });
textBox(s, "Was the evidence retrieved? Is the answer correct, relevant and supported? Correct abstention? Failure class: retrieval · context · generation · abstention", { x: 0.5, y: 4.12, w: 9, h: 0.6, fontSize: 13, color: C.text2 });
s.addNotes("Expected answers were verified against the PDF text with a verbatim quote and page number. Corpora A and B were first run at top-k 4 and 8 with similarity and MMR; corpus C (a different-domain check) at top-k 4 and 8 and with MMR. All three were then run with the V1.3 options (header stripping, hybrid, reranking, top-k 6) and with two other LLMs. Each answer was judged by reading the retrieved chunks against the expected answer; see docs/evaluation_report.md.");

// ---------------------------------------------------------------- 10. Retrieval results (chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Retrieval: was the evidence in the top-k?", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Top-k 4", labels: D.retrieval.labels, values: D.retrieval.k4 },
  { name: "Top-k 8", labels: D.retrieval.labels, values: D.retrieval.k8 },
], Object.assign({
  x: 0.5, y: 1.15, w: 5.8, h: 3.85, barDir: "col", barGrouping: "clustered",
  chartColors: [HEX.accent4, HEX.accent1], showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 11, dataLabelColor: HEX.dk1,
  showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: HEX.dk2,
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 11, valAxisLabelColor: HEX.dk2, valAxisLabelFontSize: 10,
  valAxisMaxVal: 100, valAxisMinVal: 0, valAxisLabelFormatCode: "0\"%\"", dataLabelFormatCode: "0\"%\"",
  valGridLine: { color: HEX.lt2, size: 0.75 }, catGridLine: { style: "none" },
  showTitle: true, title: "V1.1 baseline (similarity search): evidence retrieved (%)", titleFontSize: 12, titleColor: HEX.dk1,
}, chartText));
D.retrieval.notes.forEach((t, i) => {
  textBox(s, t, { x: 6.6, y: 1.25 + i * 1.25, w: 2.9, h: 1.1, fontSize: 13, color: C.text1 });
});
s.addNotes(D.retrieval.speaker);

// ---------------------------------------------------------------- 11. Answer quality (chart)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Answers: correctness and honest abstention", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Correct", labels: D.quality.labels, values: D.quality.correct },
  { name: "Partial", labels: D.quality.labels, values: D.quality.partial },
  { name: "Incorrect", labels: D.quality.labels, values: D.quality.incorrect },
], Object.assign({
  x: 0.5, y: 1.15, w: 5.8, h: 3.85, barDir: "bar", barGrouping: "stacked",
  chartColors: [HEX.accent2, HEX.accent1, HEX.accent3], showValue: true, dataLabelPosition: "ctr", dataLabelFontSize: 11, dataLabelColor: HEX.dk1,
  showLegend: true, legendPos: "b", legendFontSize: 11, legendColor: HEX.dk2,
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 11, valAxisLabelColor: HEX.dk2, valAxisLabelFontSize: 10,
  valGridLine: { color: HEX.lt2, size: 0.75 }, catGridLine: { style: "none" },
  showTitle: true, title: "V1.1 baseline (similarity search): outcomes (count)", titleFontSize: 12, titleColor: HEX.dk1,
}, chartText));
textBox(s, D.quality.abstainBig, { x: 6.6, y: 1.2, w: 2.9, h: 0.9, fontSize: 44, bold: true, color: C.accent2, fontFace: THEME.headFontFace });
textBox(s, D.quality.abstainLabel, { x: 6.6, y: 2.1, w: 2.9, h: 0.6, fontSize: 12, color: C.text2 });
textBox(s, D.quality.note, { x: 6.6, y: 2.9, w: 2.9, h: 2.0, fontSize: 13, color: C.text1 });
s.addNotes(D.quality.speaker);

// ---------------------------------------------------------------- 11b. Retrieval mode + LLM parameters (V1.2)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Similarity vs MMR search, and LLM settings", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Correct · similarity", labels: D.mmr.labels, values: D.mmr.simCorrect },
  { name: "Correct · MMR", labels: D.mmr.labels, values: D.mmr.mmrCorrect },
  { name: "Incorrect · similarity", labels: D.mmr.labels, values: D.mmr.simIncorrect },
  { name: "Incorrect · MMR", labels: D.mmr.labels, values: D.mmr.mmrIncorrect },
], Object.assign({
  x: 0.5, y: 1.15, w: 4.7, h: 2.35, barDir: "col", barGrouping: "clustered",
  chartColors: [HEX.accent2, "8FD3CA", HEX.accent3, "F2B8A7"], showValue: true, dataLabelFontSize: 9, dataLabelColor: HEX.dk1,
  showLegend: true, legendPos: "b", legendFontSize: 9, legendColor: HEX.dk2,
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 10, valAxisLabelColor: HEX.dk2, valAxisLabelFontSize: 9,
  valGridLine: { color: HEX.lt2, size: 0.75 }, catGridLine: { style: "none" },
  showTitle: true, title: "V1.2: answerable questions (A n=10, B n=6)", titleFontSize: 11, titleColor: HEX.dk1,
}, chartText));
s.addText(bullets(D.mmr.points, 11), { x: 0.5, y: 3.6, w: 4.8, h: 1.55, isTextBox: true, margin: 0, valign: "top" });
card(s, 5.5, 1.15, 4.0, 3.9, "params-card");
textBox(s, "LLM and retrieval settings", { x: 5.7, y: 1.3, w: 3.6, h: 0.35, fontSize: 16, bold: true, fontFace: THEME.headFontFace });
D.mmr.params.forEach(([k, v, why], i) => {
  const y = 1.8 + i * 0.62;
  textBox(s, k, { x: 5.7, y, w: 1.7, h: 0.3, fontSize: 12, bold: true });
  textBox(s, v, { x: 7.4, y, w: 1.9, h: 0.3, fontSize: 12, color: C.accent6, bold: true, fontFace: "Courier New" });
  textBox(s, why, { x: 5.7, y: y + 0.27, w: 3.6, h: 0.28, fontSize: 10, color: C.text2 });
});
textBox(s, "All adjustable in the sidebar; recorded in every run", { x: 5.7, y: 4.75, w: 3.6, h: 0.25, fontSize: 10, italic: true, color: C.text2 });
s.addNotes(D.mmr.speaker);

// ---------------------------------------------------------------- 12b. Improvements (V1.3)
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Improving it: hybrid search + reranker", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Correct", labels: D.v13.labels, values: D.v13.correct },
  { name: "Partial", labels: D.v13.labels, values: D.v13.partial },
  { name: "Incorrect", labels: D.v13.labels, values: D.v13.incorrect },
], Object.assign({
  x: 0.5, y: 1.15, w: 5.0, h: 2.6, barDir: "bar", barGrouping: "stacked",
  chartColors: [HEX.accent2, HEX.accent1, HEX.accent3], showValue: true, dataLabelPosition: "ctr", dataLabelFontSize: 10, dataLabelColor: HEX.dk1,
  showLegend: true, legendPos: "b", legendFontSize: 10, legendColor: HEX.dk2,
  catAxisLabelColor: HEX.dk2, catAxisLabelFontSize: 10, valAxisLabelColor: HEX.dk2, valAxisLabelFontSize: 9,
  valGridLine: { color: HEX.lt2, size: 0.75 }, catGridLine: { style: "none" },
  showTitle: true, title: "22 answerable questions, 3 corpora, top-k 4", titleFontSize: 11, titleColor: HEX.dk1,
}, chartText));
s.addText(bullets(D.v13.points, 11), { x: 0.5, y: 3.85, w: 5.0, h: 1.3, isTextBox: true, margin: 0, valign: "top" });
textBox(s, D.v13.big, { x: 5.8, y: 1.15, w: 3.7, h: 0.75, fontSize: 40, bold: true, color: C.accent2, fontFace: THEME.headFontFace });
textBox(s, D.v13.bigLabel, { x: 5.8, y: 1.9, w: 3.7, h: 0.5, fontSize: 12, color: C.text2 });
card(s, 5.8, 2.55, 3.7, 2.5, "llm-card");
textBox(s, "LLM comparison (same retrieval)", { x: 6.0, y: 2.68, w: 3.3, h: 0.3, fontSize: 13, bold: true, fontFace: THEME.headFontFace });
D.v13.llms.forEach(([m, r, n], i) => {
  const y = 3.08 + i * 0.62;
  textBox(s, m, { x: 6.0, y, w: 1.75, h: 0.28, fontSize: 11, bold: true, fontFace: "Courier New" });
  textBox(s, r, { x: 7.75, y, w: 1.6, h: 0.28, fontSize: 11, bold: true, color: C.accent6, align: "right" });
  textBox(s, n, { x: 6.0, y: y + 0.27, w: 3.35, h: 0.3, fontSize: 9.5, color: C.text2 });
});
s.addNotes(D.v13.speaker);

// ---------------------------------------------------------------- 12. Failure analysis
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Evaluation" });
s.addText("Confirmed failure cases and root causes", { placeholder: "title" });
D.failures.forEach((f, i) => {
  const x = 0.5 + i * 3.05;
  card(s, x, 1.2, 2.85, 3.75, "fail-" + i);
  textBox(s, f.cls.toUpperCase(), { x: x + 0.2, y: 1.37, w: 2.45, h: 0.25, fontSize: 10, bold: true, color: C.accent3, charSpacing: 1 });
  textBox(s, f.title, { x: x + 0.2, y: 1.65, w: 2.45, h: 0.6, fontSize: 15, bold: true, fontFace: THEME.headFontFace });
  textBox(s, f.what, { x: x + 0.2, y: 2.3, w: 2.45, h: 1.2, fontSize: 12, color: C.text2 });
  textBox(s, f.status, { x: x + 0.2, y: 3.6, w: 2.45, h: 1.25, fontSize: 12, color: C.text1 });
});
s.addNotes(D.failuresSpeaker);

// ---------------------------------------------------------------- 13. Limitations & improvements
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Conclusion" });
s.addText("Limitations and next improvements", { placeholder: "title" });
textBox(s, "Limitations", { x: 0.5, y: 1.2, w: 4.3, h: 0.35, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
s.addText(bullets(D.limitations, 13), { x: 0.5, y: 1.65, w: 4.3, h: 3.3, isTextBox: true, margin: 0, valign: "top" });
card(s, 5.2, 1.2, 4.3, 3.75, "improve-card");
textBox(s, "Improvements (evidence-based)", { x: 5.45, y: 1.35, w: 3.9, h: 0.35, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
s.addText(bullets(D.improvements, 13), { x: 5.45, y: 1.8, w: 3.85, h: 3.05, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Hybrid search, reranking and header stripping were implemented in V1.3 because the evaluation showed retrieval misses were the main failure. What is listed here is what remains, each tied to an observed failure. A larger model was tested (Llama 3.1 8B) and reads better, but it needs more memory than this 8 GB laptop.");

// ---------------------------------------------------------------- 14. Conclusion
s = pres.addSlide({ masterName: "DARK", sectionTitle: "Conclusion" });
s.addText("Conclusion", { placeholder: "title" });
s.addText(D.conclusion, { placeholder: "body" });
badge(s, 0.6, 0.7, "[✓]", C.accent2, "end-b1");
s.addNotes("Close by restating what works (grounded, cited answers; consistent abstention on unanswerable questions, on two very different kinds of documents), how evaluation drove the improvements (wrong answers 7 to 2 of 22), and what limits it now (the 3B model's reading of tables and lists). Invite questions.");

(async () => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("Wrote " + OUT);
})();

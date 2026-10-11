"""Split page Documents into overlapping chunks with stable ids."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def split_documents(pages: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Chunk each page separately so every chunk maps to exactly one page.

    Adds `chunk_index` (position within the page) and
    `chunk_id` = f"{file_hash[:12]}-p{page}-c{index}" to the page metadata.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
    )

    chunks: list[Document] = []
    for page in pages:
        file_hash = str(page.metadata.get("file_hash", ""))
        page_no = page.metadata.get("page", 0)
        index = 0
        for piece in splitter.split_text(page.page_content):
            text = piece.strip()
            if not text:
                continue
            metadata = dict(page.metadata)
            metadata["chunk_index"] = index
            metadata["chunk_id"] = f"{file_hash[:12]}-p{page_no}-c{index}"
            chunks.append(Document(page_content=text, metadata=metadata))
            index += 1
    return chunks

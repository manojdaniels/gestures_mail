from __future__ import annotations
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    chunks = splitter.split_text(text)
    # Filter out very small or empty chunks
    chunks = [c.strip() for c in chunks if c and len(c.strip()) > 50]
    return chunks
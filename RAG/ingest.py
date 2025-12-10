"""
RAG ingestion script (beginner-friendly).

What it does:
1) Reads your master resume (PDF only, text-based).
2) Splits it into overlapping chunks so retrieval works well.
3) Embeds the chunks with a small sentence-transformer model.
4) Saves everything into a local Chroma vector store on disk.

Run it:
  python ingest.py --resume /abs/path/to/my_master_resume.txt \
                   --persist-dir /abs/path/to/chroma_db
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_resume_documents(resume_path: Path) -> List[Document]:
    """Load resume content into Document objects (PDF only)."""
    if resume_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF resumes are supported in this pipeline.")
    loader = PyPDFLoader(str(resume_path))
    return loader.load()


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """
    Break long documents into chunks with overlap so the retriever
    can match smaller pieces of text to a query.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Create an embedding model (small, fast).
    This turns text into vectors that Chroma can store and search.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest_resume(resume_path: Path, persist_dir: Path) -> None:
    """End-to-end ingestion: load, chunk, embed, persist."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    raw_docs = load_resume_documents(resume_path)
    chunks = chunk_documents(raw_docs)
    embeddings = build_embeddings()

    # Clean metadata to avoid issues with some vector stores.
    filtered_chunks = filter_complex_metadata(chunks)

    Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )
    print(f"[done] Saved {len(filtered_chunks)} chunks to {persist_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest resume into Chroma.")
    parser.add_argument(
        "--resume",
        required=True,
        type=Path,
        help="Path to your master resume (PDF only).",
    )
    parser.add_argument(
        "--persist-dir",
        required=True,
        type=Path,
        help="Directory to store the Chroma DB.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_resume(resume_path=args.resume, persist_dir=args.persist_dir)


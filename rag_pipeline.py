"""
Minimal RAG helper for the Agentic Resume project.

What it does:
- Ingest your master resume (txt or pdf), chunk, embed, and persist to ChromaDB.
- Read a job description PDF, run similarity search against the resume store,
  and return the most relevant resume chunks for downstream prompting.

Usage (CLI):
  python rag_pipeline.py --job-pdf /path/to/job.pdf \
                         --resume /path/to/my_master_resume.txt \
                         --persist-dir /path/to/chroma_db \
                         --k 6

Dependencies: langchain, langchain-community, langchain-chroma,
sentence-transformers, chromadb, pypdf (via langchain-community PyPDFLoader).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_transformers import (
    LongContextReorder,
)
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def load_resume_documents(resume_path: Path) -> List[Document]:
    """Load resume content from txt or pdf into LangChain documents."""
    if resume_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(resume_path))
    elif resume_path.suffix.lower() in {".txt", ".md"}:
        loader = TextLoader(str(resume_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported resume format: {resume_path.suffix}")
    return loader.load()


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    """Chunk documents for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def build_embeddings() -> HuggingFaceEmbeddings:
    """Configure sentence-transformers embeddings."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def ingest_resume(resume_path: Path, persist_dir: Path) -> None:
    """Ingest and persist the resume vector store to disk."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    raw_docs = load_resume_documents(resume_path)
    chunks = chunk_documents(raw_docs)
    embeddings = build_embeddings()
    filtered_chunks = filter_complex_metadata(chunks)
    Chroma.from_documents(
        documents=filtered_chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )


def load_vector_store(persist_dir: Path) -> Chroma:
    """Load an existing Chroma store."""
    embeddings = build_embeddings()
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def load_job_pdf(job_pdf_path: Path) -> List[Document]:
    """Extract job description text from a PDF."""
    loader = PyPDFLoader(str(job_pdf_path))
    return loader.load()


def retrieve_resume_chunks(
    job_pdf_path: Path,
    persist_dir: Path,
    k: int = 6,
) -> List[Document]:
    """Retrieve the most relevant resume chunks for the given job PDF."""
    job_docs = load_job_pdf(job_pdf_path)
    job_text = "\n\n".join(doc.page_content for doc in job_docs)

    vector_store = load_vector_store(persist_dir)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(job_text)

    # Optional reorder to prefer broader-to-specific context in the final list.
    reorder = LongContextReorder()
    return reorder.transform_documents(results)


def demo(job_pdf: Path, resume: Path, persist_dir: Path, k: int) -> None:
    """End-to-end demo: ingest resume (if db empty) and retrieve for a job PDF."""
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        print(f"[info] Building vector store at {persist_dir}")
        ingest_resume(resume, persist_dir)
    else:
        print(f"[info] Using existing vector store at {persist_dir}")

    hits = retrieve_resume_chunks(job_pdf, persist_dir, k=k)
    print(f"[info] Retrieved {len(hits)} chunks\n")
    for i, doc in enumerate(hits, start=1):
        print(f"--- Chunk {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content[:800])
        if len(doc.page_content) > 800:
            print("... [truncated]")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAG pipeline demo.")
    parser.add_argument(
        "--job-pdf",
        required=True,
        type=Path,
        help="Path to the job description PDF.",
    )
    parser.add_argument(
        "--resume",
        required=True,
        type=Path,
        help="Path to your master resume (txt or pdf).",
    )
    parser.add_argument(
        "--persist-dir",
        required=True,
        type=Path,
        help="Directory to store/load Chroma DB.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of resume chunks to retrieve.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(job_pdf=args.job_pdf, resume=args.resume, persist_dir=args.persist_dir, k=args.k)


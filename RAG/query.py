"""
RAG query script (beginner-friendly).

What it does:
1) Reads a job description PDF.
2) Turns that text into a query.
3) Searches your Chroma vector store for the most relevant resume chunks.
4) Prints the top-K chunks so you can feed them into an LLM prompt.

Run it:
  python query.py --job-pdf /abs/path/to/job.pdf \
                  --persist-dir /abs/path/to/chroma_db \
                  --k 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def build_embeddings() -> HuggingFaceEmbeddings:
    """Same embedding model used during ingestion."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_vector_store(persist_dir: Path) -> Chroma:
    """Load the existing Chroma DB from disk."""
    embeddings = build_embeddings()
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )


def load_job_pdf(job_pdf_path: Path) -> List[Document]:
    """Extract the job description text from a PDF."""
    loader = PyPDFLoader(str(job_pdf_path))
    return loader.load()


def retrieve_resume_chunks(
    job_pdf_path: Path,
    persist_dir: Path,
    k: int = 6,
) -> List[Document]:
    """
    Retrieve the most relevant resume chunks for this job.
    - We combine all pages into one query string.
    - We ask Chroma for the top-K similar chunks.
    - Optional: reorder to place broad context before specifics.
    """
    job_docs = load_job_pdf(job_pdf_path)
    job_text = "\n\n".join(doc.page_content for doc in job_docs)

    vector_store = load_vector_store(persist_dir)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(job_text)

    reorder = LongContextReorder()
    return reorder.transform_documents(results)


def pretty_print(chunks: List[Document]) -> None:
    """Print retrieved chunks with their source info."""
    for i, doc in enumerate(chunks, start=1):
        print(f"--- Chunk {i} (source: {doc.metadata.get('source')}) ---")
        print(doc.page_content[:800])
        if len(doc.page_content) > 800:
            print("... [truncated]")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query resume RAG store.")
    parser.add_argument(
        "--job-pdf",
        required=True,
        type=Path,
        help="Path to the job description PDF.",
    )
    parser.add_argument(
        "--persist-dir",
        required=True,
        type=Path,
        help="Directory where the Chroma DB is stored.",
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
    chunks = retrieve_resume_chunks(
        job_pdf_path=args.job_pdf,
        persist_dir=args.persist_dir,
        k=args.k,
    )
    pretty_print(chunks)


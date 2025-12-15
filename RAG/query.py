"""
RAG query script (production-ready).

What it does:
1) Validates and reads a job description PDF with robust text extraction.
2) Searches the Chroma vector store for the most relevant resume chunks.
3) Validates and ranks results by relevance and quality.
4) Provides structured output with performance metrics.

Run it:
  python query.py --job-pdf /path/to/job.pdf --persist-dir /path/to/chroma_db --k 6

Features:
- Robust PDF text extraction with quality validation
- Intelligent result ranking and filtering
- Comprehensive error handling and logging
- Performance monitoring and metrics
- Resume-aware result presentation
"""

from __future__ import annotations

import os
# Disable Chroma anonymized telemetry to avoid noisy telemetry errors.
# This MUST be set before Chroma/ChromaDB is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")

import argparse
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF for better PDF processing
    HAS_PYMUPDF = True
except Exception:
    fitz = None  # type: ignore
    HAS_PYMUPDF = False


# Configuration constants
MAX_JOB_FILE_SIZE_MB = 5
MIN_JOB_CONTENT_LENGTH = 200
MAX_RETRIEVE_K = 20
DEFAULT_K = 6


def validate_job_file(job_pdf_path: Path) -> None:
    """Validate job description PDF before processing."""
    if not job_pdf_path.exists():
        raise FileNotFoundError(f"Job PDF not found: {job_pdf_path}")

    if not job_pdf_path.is_file():
        raise ValueError(f"Path is not a file: {job_pdf_path}")

    if job_pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF files are supported. Got: {job_pdf_path.suffix}")

    # Check file size
    file_size_mb = job_pdf_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_JOB_FILE_SIZE_MB:
        raise ValueError(f"Job PDF too large: {file_size_mb:.1f}MB (max: {MAX_JOB_FILE_SIZE_MB}MB)")

    # Basic PDF validation
    try:
        if HAS_PYMUPDF:
            assert fitz is not None
            with fitz.open(str(job_pdf_path)) as doc:
                if len(doc) == 0:
                    raise ValueError("Job PDF appears to be empty (no pages)")
        else:
            reader = PdfReader(str(job_pdf_path))
            if not reader.pages:
                raise ValueError("Job PDF appears to be empty (no pages)")
    except Exception as e:
        raise ValueError(f"Invalid or corrupted job PDF: {e}")


def extract_job_text_with_pymupdf(job_pdf_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from job description PDF using PyMuPDF with quality validation.

    Similar to resume extraction but with job-specific validation:
    - Checks for job-related keywords to validate content type
    - Uses different minimum length thresholds
    - Maintains page-by-page extraction for robustness
    """
    # Initialize metadata tracking
    metadata = {"source": str(job_pdf_path), "pages": 0, "total_chars": 0}

    try:
        text_parts: List[str] = []

        if HAS_PYMUPDF:
            assert fitz is not None
            with fitz.open(str(job_pdf_path)) as doc:
                metadata["pages"] = len(doc)

                # Extract text from each page
                for page_num, page in enumerate(doc):
                    try:
                        page_text = page.get_text()
                        if page_text.strip():  # Skip empty pages
                            text_parts.append(page_text)
                            logger.debug(f"Extracted {len(page_text)} chars from job PDF page {page_num + 1}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from job PDF page {page_num + 1}: {e}")
                        continue
        else:
            reader = PdfReader(str(job_pdf_path))
            metadata["pages"] = len(reader.pages)
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
                        logger.debug(f"Extracted {len(page_text)} chars from job PDF page {page_num + 1} (pypdf)")
                except Exception as e:
                    logger.warning(f"Failed to extract text from job PDF page {page_num + 1} (pypdf): {e}")
                    continue

        # Combine all pages
        full_text = "\n\n".join(text_parts)
        metadata["total_chars"] = len(full_text)

        # Quality validation for job descriptions
        # 1. Minimum content length
        if len(full_text.strip()) < MIN_JOB_CONTENT_LENGTH:
            raise ValueError(f"Job description too short: {len(full_text)} chars (min: {MIN_JOB_CONTENT_LENGTH})")

        # 2. Validate this looks like a job description
        job_keywords = ['requirements', 'responsibilities', 'qualifications', 'experience', 'skills']
        keyword_count = sum(1 for keyword in job_keywords if keyword.lower() in full_text.lower())

        if keyword_count < 2:
            logger.warning(f"Job PDF may not be a valid job description (found {keyword_count} job-related keywords)")

        return full_text, metadata

    except Exception as e:
        logger.error(f"Failed to extract text from job PDF: {e}")
        raise


def load_job_pdf(job_pdf_path: Path) -> List[Document]:
    """Load and validate job description PDF with enhanced extraction."""
    logger.info(f"Loading job description: {job_pdf_path}")

    # Validate file
    validate_job_file(job_pdf_path)

    # Extract text with quality validation
    start_time = time.time()
    try:
        text_content, metadata = extract_job_text_with_pymupdf(job_pdf_path)
        extraction_time = time.time() - start_time
        logger.info(f"Job text extraction completed in {extraction_time:.2f}s - {metadata['total_chars']} chars from {metadata['pages']} pages")

        # Create document with rich metadata
        doc = Document(
            page_content=text_content,
            metadata={
                **metadata,
                "extraction_time": extraction_time,
                "content_type": "job_description",
                "query_timestamp": time.time()
            }
        )

        return [doc]

    except Exception as e:
        logger.error(f"Failed to load job PDF: {e}")
        raise


def retrieve_resume_chunks_from_text(
    job_text: str,
    persist_dir: Path,
    k: int = DEFAULT_K,
    job_source: str = "job_text",
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Retrieve resume chunks given raw job description text.

    This is the same retrieval pipeline as `retrieve_resume_chunks`, but skips PDF parsing.
    Useful for manual testing, interactive usage, and programmatic evaluation.
    """
    query_stats = {
        "start_time": time.time(),
        "job_extraction_time": 0,
        "vector_search_time": 0,
        "validation_time": 0,
        "total_chunks_retrieved": 0,
        "chunks_returned": 0,
        "success": False,
    }

    try:
        logger.info("=" * 50)
        logger.info("STARTING RESUME CHUNK RETRIEVAL")
        logger.info("=" * 50)

        k = validate_retrieval_params(k)
        logger.info(f"Retrieval parameters: k={k}")

        if not job_text or not job_text.strip():
            raise ValueError("Job description text is empty")
        if len(job_text.strip()) < MIN_JOB_CONTENT_LENGTH:
            logger.warning(
                f"Job description text is short ({len(job_text.strip())} chars). "
                f"Results may be less reliable (PDF min is {MIN_JOB_CONTENT_LENGTH})."
            )

        vector_store = load_vector_store(persist_dir)

        # Figure out how many docs exist to avoid over-requesting results.
        doc_count = vector_store._collection.count() if hasattr(vector_store, "_collection") else 0
        search_k = min(max(k * 2, 10), max(doc_count, 1))

        def _normalize_vector_score(raw_score: float) -> float:
            """
            Normalize different score conventions into a stable [0, 1] relevance score.

            - If it looks like cosine similarity in [-1, 1], map to [0, 1].
            - If it looks like a distance (>= 0), convert to relevance via 1/(1+distance).
            - Otherwise clamp as a safe fallback.
            """
            try:
                s = float(raw_score)
            except Exception:
                return 0.0

            if -1.0 <= s <= 1.0:
                return (s + 1.0) / 2.0
            if s >= 0.0:
                return 1.0 / (1.0 + s)
            return 0.0

        # Prefer vector-store scores when available (normalized to [0,1]).
        search_start = time.time()
        raw_docs: List[Document]
        try:
            # This API exists across many vectorstores and avoids [0,1] assumptions.
            docs_and_scores = vector_store.similarity_search_with_score(job_text, k=search_k)
            raw_docs = []
            for doc, score in docs_and_scores:
                doc.metadata["vector_score_raw"] = float(score)
                doc.metadata["vector_relevance_score"] = _normalize_vector_score(float(score))
                raw_docs.append(doc)
        except Exception:
            # Fall back to plain similarity search if relevance-score API isn't available.
            raw_docs = vector_store.similarity_search(job_text, k=search_k)

        query_stats["vector_search_time"] = time.time() - search_start
        query_stats["total_chunks_retrieved"] = len(raw_docs)

        reorder = LongContextReorder()
        reordered_results = reorder.transform_documents(raw_docs)

        validation_start = time.time()
        final_results = validate_and_filter_results(reordered_results, job_text, k)
        query_stats["validation_time"] = time.time() - validation_start
        query_stats["chunks_returned"] = len(final_results)

        query_stats["total_time"] = time.time() - query_stats["start_time"]
        query_stats["success"] = True

        # Attach minimal job metadata to chunks for downstream inspection
        for chunk in final_results:
            chunk.metadata.setdefault("job_source", job_source)

        logger.info("=" * 50)
        logger.info("RETRIEVAL COMPLETED SUCCESSFULLY")
        logger.info(f"Retrieved: {query_stats['total_chunks_retrieved']} → Filtered: {query_stats['chunks_returned']}")
        logger.info(f"Total time: {query_stats['total_time']:.2f}s")
        logger.info("=" * 50)

        return final_results, query_stats

    except Exception as e:
        query_stats["total_time"] = time.time() - query_stats["start_time"]
        query_stats["error"] = str(e)
        logger.error("=" * 50)
        logger.error("RETRIEVAL FAILED")
        logger.error(f"Error: {e}")
        logger.error(f"Total time: {query_stats['total_time']:.2f}s")
        logger.error("=" * 50)
        raise


def build_embeddings() -> HuggingFaceEmbeddings:
    """Create embedding model with validation."""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def load_vector_store(persist_dir: Path) -> Chroma:
    """
    Load and validate the existing Chroma vector database from disk.

    Performs several validation checks:
    1. Directory existence and accessibility
    2. Chroma database initialization with embeddings
    3. Content validation (non-empty database)
    4. Logging of database statistics
    """
    # Validate directory exists and is accessible
    if not persist_dir.exists():
        raise FileNotFoundError(f"Vector store directory not found: {persist_dir}")

    if not persist_dir.is_dir():
        raise ValueError(f"Path is not a directory: {persist_dir}")

    try:
        # Initialize embeddings (must match ingestion embeddings)
        embeddings = build_embeddings()
        client_settings = Settings(anonymized_telemetry=False)

        # Load persisted Chroma database
        vector_store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            client_settings=client_settings,
        )

        # Validate database contains documents
        doc_count = vector_store._collection.count() if hasattr(vector_store, '_collection') else 0
        if doc_count == 0:
            raise ValueError(f"Vector store is empty: {persist_dir}")

        logger.info(f"Loaded vector store with {doc_count} documents")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise


def validate_retrieval_params(k: int) -> int:
    """
    Validate and adjust retrieval parameters within safe bounds.

    Ensures k (number of chunks to retrieve) is within reasonable limits:
    - Minimum: 1 (must return at least one result)
    - Maximum: MAX_RETRIEVE_K (prevent excessive computation/memory usage)

    Returns validated k value.
    """
    if k < 1:
        logger.warning(f"Invalid k value {k}, setting to 1")
        return 1
    elif k > MAX_RETRIEVE_K:
        logger.warning(f"k value {k} too high, limiting to {MAX_RETRIEVE_K}")
        return MAX_RETRIEVE_K
    return k


def calculate_relevance_score(chunk: Document, job_text: str) -> float:
    """
    Calculate a relevance score for a resume chunk based on job description.

    Uses a multi-factor scoring approach:
    1. Jaccard similarity: Measures overlap between job and resume vocabulary
    2. Section bonus: Prioritizes relevant resume sections (skills > experience > others)
    3. Length bonus: Slight preference for more substantial chunks

    Returns score between 0.0 and 1.0, where higher is more relevant.
    """
    chunk_text = chunk.page_content.lower()
    job_lower = job_text.lower()

    # Extract word sets for vocabulary overlap analysis
    job_words = set(re.findall(r'\b\w+\b', job_lower))
    chunk_words = set(re.findall(r'\b\w+\b', chunk_text))

    # Calculate Jaccard similarity (intersection over union)
    # This measures how much of the job's vocabulary appears in the chunk
    intersection = len(job_words.intersection(chunk_words))
    union = len(job_words.union(chunk_words))

    if union == 0:
        return 0.0

    jaccard = intersection / union

    # Apply section-specific relevance bonuses
    # Skills are most valuable for job matching, followed by experience
    section_type = chunk.metadata.get('section_type', 'other')
    section_bonus = {
        'experience': 0.1,    # Work history is highly relevant
        'skills': 0.15,       # Technical skills are most important
        'projects': 0.05,     # Portfolio shows practical application
        'education': 0.05     # Educational background provides context
    }.get(section_type, 0.0)

    # Small bonus for longer chunks (up to 10% bonus for 1000+ chars)
    # Prefers substantial content over brief fragments
    length_bonus = min(len(chunk.page_content) / 1000.0, 0.1)

    # Combine all factors, capped at 1.0
    return min(jaccard + section_bonus + length_bonus, 1.0)


def validate_and_filter_results(results: List[Document], job_text: str, k: int) -> List[Document]:
    """
    Validate and filter retrieval results by relevance scoring.

    Applies post-retrieval filtering to improve result quality:
    1. Scores all retrieved chunks for job relevance
    2. Sorts by relevance score (highest first)
    3. Filters out very low-relevance results
    4. Ensures minimum results returned (k/2 or at least 1)
    5. Returns top-k most relevant chunks
    """
    if not results:
        raise ValueError("No results retrieved from vector store")

    # Calculate relevance scores for all retrieved chunks
    scored_results = []
    for chunk in results:
        lexical_score = calculate_relevance_score(chunk, job_text)
        vector_score = chunk.metadata.get("vector_relevance_score", None)

        # Prefer vector score when present; otherwise fall back to lexical.
        if isinstance(vector_score, (int, float)):
            base_score = float(vector_score)
        else:
            base_score = float(lexical_score)

        # Penalize low-signal sections so they don't crowd out top-k.
        # We still keep the min-results fallback below, so we won't return nothing.
        section_type = str(chunk.metadata.get("section_type", "other")).strip().lower()
        section_penalty = {
            "contact": 0.05,
            "other": 0.70,
        }.get(section_type, 1.0)
        final_score = base_score * section_penalty

        chunk.metadata["lexical_relevance_score"] = float(lexical_score)
        chunk.metadata["relevance_score_raw"] = float(base_score)
        chunk.metadata["relevance_score"] = float(final_score)
        scored_results.append((chunk, final_score))

    # Sort by relevance score (highest relevance first)
    scored_results.sort(key=lambda x: x[1], reverse=True)

    # Apply quality filtering with fallback guarantees
    # Keep chunks with relevance > 5%, but ensure we return at least k/2 results
    min_results = max(k // 2, 1)
    filtered_results = []

    for chunk, score in scored_results:
        # Include if score is good enough OR we haven't met minimum threshold
        if score > 0.05 or len(filtered_results) < min_results:
            filtered_results.append(chunk)

        # Stop once we have enough results
        if len(filtered_results) >= k:
            break

    # Final validation and fallback
    if not filtered_results:
        logger.warning("All results filtered out due to low relevance")
        return results[:k]  # Return original top-k as fallback

    # Log statistics for monitoring
    scores = [chunk.metadata['relevance_score'] for chunk in filtered_results]
    logger.info(f"Result validation: {len(filtered_results)} chunks, avg relevance: {sum(scores)/len(scores):.3f}")

    return filtered_results


def retrieve_resume_chunks(
    job_pdf_path: Path,
    persist_dir: Path,
    k: int = DEFAULT_K,
) -> Tuple[List[Document], Dict[str, Any]]:
    """
    Retrieve and validate the most relevant resume chunks for a job description.

    Retrieval pipeline:
    1. Extract and validate job description text
    2. Load persisted vector store from disk
    3. Perform semantic search (retrieve 2x requested chunks)
    4. Reorder results for optimal context flow
    5. Filter and rank by relevance scoring
    6. Return top-k most relevant chunks with performance stats

    Returns:
        Tuple of (filtered_chunks, performance_stats)
    """
    # Initialize comprehensive performance tracking
    query_stats = {
        "start_time": time.time(),
        "job_extraction_time": 0,
        "vector_search_time": 0,
        "validation_time": 0,
        "total_chunks_retrieved": 0,
        "chunks_returned": 0,
        "success": False
    }

    try:
        logger.info("=" * 50)
        logger.info("STARTING RESUME CHUNK RETRIEVAL")
        logger.info("=" * 50)

        # Validate and adjust retrieval parameters
        k = validate_retrieval_params(k)
        logger.info(f"Retrieval parameters: k={k}")

        # Stage 1: Load and extract job description
        job_load_start = time.time()
        job_docs = load_job_pdf(job_pdf_path)
        job_text = "\n\n".join(doc.page_content for doc in job_docs)
        query_stats["job_extraction_time"] = time.time() - job_load_start

        # Delegate to text-based retrieval for consistent behavior (and better testability).
        final_results, stats = retrieve_resume_chunks_from_text(
            job_text=job_text,
            persist_dir=persist_dir,
            k=k,
            job_source=str(job_pdf_path),
        )

        # Merge stats (keep job_extraction_time from this function)
        stats["job_extraction_time"] = query_stats["job_extraction_time"]
        return final_results, stats

    except Exception as e:
        # Capture error details for debugging
        query_stats["total_time"] = time.time() - query_stats["start_time"]
        query_stats["error"] = str(e)

        logger.error("=" * 50)
        logger.error("RETRIEVAL FAILED")
        logger.error(f"Error: {e}")
        logger.error(f"Total time: {query_stats['total_time']:.2f}s")
        logger.error("=" * 50)

        raise


def pretty_print(chunks: List[Document], stats: Optional[Dict[str, Any]] = None) -> None:
    """
    Print retrieved chunks with rich metadata and performance statistics.

    Displays results in a human-readable format with:
    - Performance metrics (timing, counts, stages)
    - Chunk metadata (section type, relevance score, source)
    - Content previews (truncated for readability)
    - Clear visual separation between chunks
    """
    # Display performance statistics if available
    if stats:
        print(f"\n📊 Query Performance Stats:")
        print(f"   Total time: {stats['total_time']:.2f}s")
        print(f"   Chunks retrieved: {stats['total_chunks_retrieved']}")
        print(f"   Chunks returned: {stats['chunks_returned']}")
        print(f"   Job extraction: {stats['job_extraction_time']:.2f}s")
        print(f"   Vector search: {stats['vector_search_time']:.2f}s")
        print(f"   Validation: {stats['validation_time']:.2f}s")
        print()

    print(f"📄 Retrieved {len(chunks)} Resume Chunks:\n")

    for i, doc in enumerate(chunks, start=1):
        metadata = doc.metadata

        # Build informative header showing chunk metadata
        section = metadata.get('section_type', 'unknown')
        relevance = metadata.get('relevance_score', 0.0)
        source = Path(metadata.get('source', 'unknown')).name

        header = f"--- Chunk {i} | Section: {section} | Relevance: {relevance:.3f} | Source: {source} ---"
        print(header)

        # Display content preview (truncated for readability)
        content = doc.page_content.strip()
        preview_length = 800

        if len(content) <= preview_length:
            print(content)
        else:
            print(content[:preview_length])
            print(f"... [truncated, {len(content) - preview_length} more chars]")

        print()


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure structured logging."""
    # Remove default handler
    logger.remove()

    # Console handler with colors
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # File handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="1 week"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query resume RAG store (production-ready).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query.py --job-pdf ./job_description.pdf --persist-dir ./chroma_db
  python query.py --job-pdf ./job.pdf --persist-dir ./db --k 8 --log-level DEBUG
  python query.py --job-pdf ./job.pdf --persist-dir ./db --log-file ./query.log
        """
    )
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
        help="Directory where the Chroma vector database is stored.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of resume chunks to retrieve (default: {DEFAULT_K}, max: {MAX_RETRIEVE_K}).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional log file path for detailed logging.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    try:
        # Run query with performance monitoring
        chunks, stats = retrieve_resume_chunks(
            job_pdf_path=args.job_pdf,
            persist_dir=args.persist_dir,
            k=args.k,
        )

        # Display results
        pretty_print(chunks, stats)

        # Exit successfully
        exit(0)

    except KeyboardInterrupt:
        logger.warning("Query interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Query failed with error: {e}")
        exit(1)


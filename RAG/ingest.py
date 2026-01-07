"""
RAG ingestion script (production-ready).

What it does:
1) Validates and reads your master resume (PDF format).
2) Applies resume-aware chunking with rich metadata.
3) Embeds chunks with quality validation.
4) Saves to Chroma vector store with comprehensive error handling.

Run it:
  python ingest.py --resume /path/to/resume.pdf --persist-dir /path/to/chroma_db

Features:
- Multi-format PDF support with robust text extraction
- Resume-aware chunking (preserves sections and bullet points)
- Comprehensive validation and error handling
- Structured logging and performance monitoring
- Rich metadata for better retrieval
"""

from __future__ import annotations

import os
# Disable Chroma telemetry (prevents PostHog capture() errors).
# Must be set before Chroma/ChromaDB is imported.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")

import argparse
import re
import time
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF for better PDF processing
    HAS_PYMUPDF = True
except Exception:
    fitz = None  # type: ignore
    HAS_PYMUPDF = False

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Configuration constants
MAX_FILE_SIZE_MB = 10
MIN_CONTENT_LENGTH = 500
MAX_CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 100
CHUNK_OVERLAP = 120

# Resume section patterns for intelligent chunking
# These regex patterns help identify different sections in a resume
# Used to apply different chunking strategies based on content type
RESUME_SECTIONS = {
    'contact': re.compile(r'^(contact|personal|information)', re.IGNORECASE),
    'summary': re.compile(r'^(summary|objective|profile)', re.IGNORECASE),
    'experience': re.compile(r'^(experience|work|work history|employment|professional)', re.IGNORECASE),
    'education': re.compile(r'^(education|academic|degree|university)', re.IGNORECASE),
    'skills': re.compile(r'^(skills|technologies|technical|competencies)', re.IGNORECASE),
    'projects': re.compile(r'^(projects|portfolio|achievements)', re.IGNORECASE),
    'certifications': re.compile(r'^(certifications|licenses|awards)', re.IGNORECASE),
}


def validate_file(resume_path: Path) -> None:
    """
    Validate resume file before processing.

    Performs multiple validation checks:
    1. File existence and type validation
    2. File size limits to prevent memory issues
    3. PDF integrity validation using PyMuPDF
    """
    # Check if file exists and is accessible
    if not resume_path.exists():
        raise FileNotFoundError(f"Resume file not found: {resume_path}")

    if not resume_path.is_file():
        raise ValueError(f"Path is not a file: {resume_path}")

    # Ensure it's a PDF file (only format we support)
    if resume_path.suffix.lower() != ".pdf":
        raise ValueError(f"Only PDF resumes are supported. Got: {resume_path.suffix}")

    # Check file size to prevent memory issues with large files
    file_size_mb = resume_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")

    # Basic PDF validation - ensure file can be opened and has content
    try:
        if HAS_PYMUPDF:
            assert fitz is not None
            with fitz.open(str(resume_path)) as doc:
                if len(doc) == 0:
                    raise ValueError("PDF appears to be empty (no pages)")
        else:
            reader = PdfReader(str(resume_path))
            if not reader.pages:
                raise ValueError("PDF appears to be empty (no pages)")
    except Exception as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")


def extract_text_with_pymupdf(resume_path: Path) -> Tuple[str, Dict[str, Any]]:
    """
    Extract text from PDF using PyMuPDF with quality validation.

    Uses PyMuPDF (fitz) for robust PDF text extraction that handles:
    - Multi-column layouts
    - Various PDF formats and encodings
    - OCR-generated text

    Returns:
        Tuple of (extracted_text, metadata_dict)
    """
    # Initialize metadata tracking
    metadata = {"source": str(resume_path), "pages": 0, "total_chars": 0}

    def _extract_with_pymupdf() -> Tuple[str, int]:
        if not HAS_PYMUPDF:
            return "", 0
        assert fitz is not None
        with fitz.open(str(resume_path)) as doc:
            pages = len(doc)
            parts: List[str] = []
            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text()
                    if page_text.strip():
                        parts.append(page_text)
                        logger.debug(f"Extracted {len(page_text)} chars from page {page_num + 1} (pymupdf)")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num + 1} (pymupdf): {e}")
                    continue
            return "\n\n".join(parts), pages

    def _extract_with_pypdf() -> Tuple[str, int]:
        reader = PdfReader(str(resume_path))
        pages = len(reader.pages)
        parts: List[str] = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(page_text)
                    logger.debug(f"Extracted {len(page_text)} chars from page {page_num + 1} (pypdf)")
            except Exception as e:
                logger.warning(f"Failed to extract text from page {page_num + 1} (pypdf): {e}")
                continue
        return "\n\n".join(parts), pages

    def _extract_with_pdftotext() -> Tuple[str, int]:
        """
        Best-effort extraction using the system `pdftotext` tool (if installed).
        This often works better on certain PDFs where libraries struggle.
        """
        try:
            # `-` outputs to stdout. `-layout` preserves columns/spacing better.
            proc = subprocess.run(
                ["pdftotext", "-layout", "-nopgbrk", str(resume_path), "-"],
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return "", 0

        out = (proc.stdout or "").strip()
        # Page count isn't available here; keep 0 so we can prefer library page counts.
        return out, 0

    try:
        # Try multiple extractors and pick the best result by length.
        candidates: List[Tuple[str, int, str]] = []
        txt_pm, pages_pm = _extract_with_pymupdf()
        if txt_pm.strip():
            candidates.append((txt_pm, pages_pm, "pymupdf"))
        txt_pp, pages_pp = _extract_with_pypdf()
        if txt_pp.strip():
            candidates.append((txt_pp, pages_pp, "pypdf"))
        txt_pt, pages_pt = _extract_with_pdftotext()
        if txt_pt.strip():
            candidates.append((txt_pt, pages_pt, "pdftotext"))

        if not candidates:
            raise ValueError("All PDF text extraction methods returned empty text")

        # Pick the longest extracted text (simple, effective heuristic).
        full_text, pages, method = max(candidates, key=lambda x: len(x[0]))

        # Prefer a non-zero page count if available.
        metadata["pages"] = pages if pages > 0 else max(p for _, p, _ in candidates) if candidates else 0
        metadata["extraction_method"] = method

        metadata["total_chars"] = len(full_text)

        # Quality validation checks
        # 1. Minimum content length to ensure meaningful extraction
        if len(full_text.strip()) < MIN_CONTENT_LENGTH:
            raise ValueError(f"Extracted text too short: {len(full_text)} chars (min: {MIN_CONTENT_LENGTH})")

        # 2. Check for gibberish content (common with failed OCR or corrupted PDFs)
        alpha_ratio = len(re.findall(r'[a-zA-Z0-9]', full_text)) / len(full_text) if full_text else 0
        if alpha_ratio < 0.3:
            logger.warning(f"Low alphanumeric ratio ({alpha_ratio:.2f}) - possible OCR or extraction issues")

        return full_text, metadata

    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise


def load_resume_documents(resume_path: Path) -> List[Document]:
    """Load and validate resume content into Document objects with enhanced PDF processing."""
    logger.info(f"Starting resume ingestion: {resume_path}")

    # Validate file
    validate_file(resume_path)

    # Extract text with quality validation
    start_time = time.time()
    try:
        text_content, metadata = extract_text_with_pymupdf(resume_path)
        extraction_time = time.time() - start_time
        logger.info(f"Text extraction completed in {extraction_time:.2f}s - {metadata['total_chars']} chars from {metadata['pages']} pages")

        # Create document with rich metadata
        doc = Document(
            page_content=text_content,
            metadata={
                **metadata,
                "extraction_time": extraction_time,
                "content_type": "resume",
                "ingestion_timestamp": time.time()
            }
        )

        return [doc]

    except Exception as e:
        logger.error(f"Failed to load resume documents: {e}")



def identify_section(text: str) -> str:
    """
    Identify which resume section a piece of text belongs to.

    Uses the first line of text to match against predefined section patterns.
    This helps apply different chunking strategies based on content type.

    Returns:
        Section name (e.g., 'experience', 'skills') or 'other' if no match
    """
    # Consider the first few non-empty lines; some PDFs omit clean header line breaks.
    lines = [ln.strip() for ln in text.strip().split("\n") if ln.strip()]
    header_probe = "\n".join(lines[:5]) if lines else ""
    first_line = lines[0] if lines else ""

    # Check against all defined section patterns
    for section_name, pattern in RESUME_SECTIONS.items():
        if pattern.search(first_line) or pattern.search(header_probe):
            return section_name

    # Default to 'other' for unrecognized sections
    return "other"


def clean_resume_text(text: str) -> str:
    """
    Clean and normalize resume text for better chunking.

    Performs several normalization steps:
    1. Reduces excessive whitespace and newlines
    2. Standardizes bullet point markers
    3. Removes page numbers and page breaks
    """
    # Remove excessive whitespace - consolidate multiple newlines and spaces
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r' {2,}', ' ', text)      # Max 1 consecutive space

    # Normalize bullet points - convert various bullet symbols to standard •
    text = re.sub(r'^[•●○◦]', '•', text, flags=re.MULTILINE)

    # Remove page numbers and page breaks that might interfere with chunking
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Line with only numbers

    return text.strip()


def resume_aware_chunking(docs: List[Document]) -> List[Document]:
    """
    Implement intelligent resume-aware chunking strategy.

    This function applies different chunking strategies based on resume section types:
    - Skills: Small chunks with high overlap for precise keyword matching
    - Experience/Projects: Large chunks with standard overlap to preserve narrative context
    - Other sections: Medium chunks with standard overlap

    The strategy aims to optimize retrieval quality by matching chunk size to content type.
    """
    logger.info("Starting resume-aware chunking")

    all_chunks = []
    start_time = time.time()

    for doc in docs:
        # Clean the text first to remove formatting artifacts
        cleaned_text = clean_resume_text(doc.page_content)

        def split_into_sections(text: str) -> List[str]:
            """
            Split resume text into sections while preserving headers.
            Handles PDFs that include leading spaces or 'HEADER:' formats.
            """
            # 1) Try ALL CAPS header heuristic first.
            candidate = re.split(r'\n(?=[A-Z][^a-z]*\n)', text)
            candidate = [s for s in candidate if s and s.strip()]
            if len(candidate) > 1:
                return candidate

            # 2) Robust heading-based split (preserve heading line with its content).
            heading_words = [
                "CONTACT INFORMATION", "CONTACT",
                "SUMMARY", "PROFESSIONAL SUMMARY", "OBJECTIVE", "PROFILE",
                "EXPERIENCE", "WORK HISTORY", "WORK EXPERIENCE", "PROFESSIONAL EXPERIENCE", "EMPLOYMENT",
                "EDUCATION",
                "SKILLS", "TECHNICAL SKILLS", "TECHNOLOGIES",
                "PROJECTS",
                "CERTIFICATIONS", "CERTIFICATES", "AWARDS",
            ]
            heading_pat = re.compile(
                r"(?im)^\s*(?P<h>(" + "|".join(re.escape(h) for h in heading_words) + r"))\s*:?\s*$"
            )
            matches = list(heading_pat.finditer(text))
            if not matches:
                return [text]

            parts: List[str] = []
            for i, m in enumerate(matches):
                start = m.start()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                section = text[start:end].strip()
                if section:
                    parts.append(section)

            # If there is a prefix before the first heading, keep it as its own section.
            prefix = text[: matches[0].start()].strip()
            if prefix:
                parts.insert(0, prefix)
            return parts

        sections = split_into_sections(cleaned_text)

        for section_text in sections:
            if len(section_text.strip()) < MIN_CHUNK_SIZE:
                continue

            # Identify section type to apply appropriate chunking strategy
            section_type = identify_section(section_text)

            # Different chunking strategies based on section type
            if section_type == "skills":
                # Skills: smaller chunks, more overlap for precise matching
                # Small chunks help match specific technical skills
                # High overlap ensures skills don't get separated
                chunk_size = 300
                overlap = 80
            elif section_type in ["experience", "projects"]:
                # Experience/Projects: larger chunks to preserve context
                # Large chunks maintain narrative flow of work experience
                chunk_size = MAX_CHUNK_SIZE
                overlap = CHUNK_OVERLAP
            else:
                # Other sections: standard chunking
                # Balanced approach for education, summary, etc.
                chunk_size = 600
                overlap = CHUNK_OVERLAP

            # Create section-specific splitter with hierarchical separators
            # Tries to split on paragraph breaks first, then sentences, then words
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""],  # Hierarchical splitting
                keep_separator=True  # Preserve formatting in splits
            )

            # Split the section into chunks
            section_chunks = splitter.split_text(section_text)

            # Create Document objects with rich metadata for each chunk
            for i, chunk_text in enumerate(section_chunks):
                if len(chunk_text.strip()) < MIN_CHUNK_SIZE:
                    continue

                # Enrich metadata with chunking information
                chunk_metadata = {
                    **doc.metadata,  # Preserve original document metadata
                    "section_type": section_type,  # Section classification
                    "chunk_index": i,  # Position within section
                    "total_chunks_in_section": len(section_chunks),  # Section size info
                    "chunk_size": len(chunk_text),  # Actual chunk size
                    "is_resume_chunk": True,  # Flag for resume-specific processing
                    "chunking_strategy": "resume_aware"  # Strategy identifier
                }

                chunk_doc = Document(
                    page_content=chunk_text.strip(),
                    metadata=chunk_metadata
                )

                all_chunks.append(chunk_doc)

    chunking_time = time.time() - start_time
    logger.info(f"Resume-aware chunking completed in {chunking_time:.2f}s - created {len(all_chunks)} chunks")

    return all_chunks


def chunk_documents(
    docs: List[Document],
    use_resume_aware: bool = True,
) -> List[Document]:
    """
    Break documents into chunks using resume-aware strategy.

    This is the main entry point for document chunking. By default, uses
    intelligent resume-aware chunking that adapts to different section types.
    Falls back to basic chunking if resume-aware is disabled.

    Args:
        docs: Input documents to be chunked
        use_resume_aware: Whether to use intelligent resume-aware chunking
                         (recommended for best retrieval quality)
    """
    if use_resume_aware:
        # Use intelligent chunking optimized for resume structure
        return resume_aware_chunking(docs)
    else:
        # Fallback to basic chunking - less optimal but faster
        logger.warning("Using basic chunking strategy - resume-aware recommended")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Fixed size for all content
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],  # Standard separators only
        )
        return splitter.split_documents(docs)


def build_embeddings() -> HuggingFaceEmbeddings:
    """
    Create and validate the sentence transformer embedding model.

    Uses all-MiniLM-L6-v2 which provides:
    - Good balance of quality and speed (384 dimensions)
    - Strong semantic understanding for sentence similarity
    - Normalized embeddings for consistent similarity scores
    - CPU-only for deployment consistency
    """
    try:
        # Initialize sentence transformer model for semantic embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Efficient general-purpose model
            model_kwargs={'device': 'cpu'},  # Explicitly use CPU for consistency across environments
            encode_kwargs={'normalize_embeddings': True}  # L2 normalization for cosine similarity
        )
        logger.info("Embedding model loaded successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


def validate_chunks(chunks: List[Document]) -> List[Document]:
    """
    Validate chunk quality and content to ensure retrieval quality.

    Performs several quality checks:
    1. Minimum content length validation
    2. Gibberish detection (high special character ratio)
    3. Metadata completeness validation
    4. Statistical reporting
    """
    if not chunks:
        raise ValueError("No chunks generated from resume")

    valid_chunks = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        content = chunk.page_content.strip()

        # Skip empty or too-short chunks that won't be useful for retrieval
        if len(content) < MIN_CHUNK_SIZE:
            logger.debug(f"Skipping chunk {i}: too short ({len(content)} chars)")
            continue

        # Check for gibberish content (extraction artifacts, symbols, etc.)
        # High ratio of special characters often indicates PDF parsing issues
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s]', content))
        if special_chars / len(content) > 0.5:
            logger.warning(f"Chunk {i} has high special character ratio - possible extraction issue")
            continue

        # Ensure required metadata exists for downstream processing
        if not chunk.metadata.get('section_type'):
            chunk.metadata['section_type'] = identify_section(content)

        total_chars += len(content)
        valid_chunks.append(chunk)

    # Ensure we have at least some valid chunks
    if not valid_chunks:
        raise ValueError("All chunks failed validation")

    logger.info(f"Chunk validation complete: {len(valid_chunks)} valid chunks, {total_chars} total chars")
    return valid_chunks


def validate_vector_store(persist_dir: Path, expected_chunks: int) -> bool:
    """Validate that vector store was created successfully."""
    try:
        # Try to load the vector store
        embeddings = build_embeddings()
        client_settings = Settings(anonymized_telemetry=False)
        vector_store = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embeddings,
            client_settings=client_settings,
        )

        # Basic health check
        count = vector_store._collection.count() if hasattr(vector_store, '_collection') else 0

        if count != expected_chunks:
            logger.warning(f"Vector store count mismatch: expected {expected_chunks}, got {count}")
            return False

        # Test a simple search
        test_results = vector_store.similarity_search("test", k=1)
        if not test_results:
            logger.warning("Vector store search test failed")
            return False

        logger.info(f"Vector store validation successful: {count} documents indexed")
        return True

    except Exception as e:
        logger.error(f"Vector store validation failed: {e}")
        return False


def ingest_resume(resume_path: Path, persist_dir: Path) -> Dict[str, Any]:
    """
    End-to-end resume ingestion pipeline with comprehensive monitoring.

    Pipeline stages:
    1. Document loading and validation
    2. Intelligent chunking with resume-aware strategies
    3. Quality validation and filtering
    4. Embedding generation
    5. Vector store creation and persistence
    6. Final validation and statistics

    Returns detailed performance statistics for monitoring and debugging.
    """
    # Initialize comprehensive performance tracking
    ingestion_stats = {
        "start_time": time.time(),
        "success": False,
        "chunks_created": 0,
        "embedding_time": 0,
        "total_time": 0
    }

    try:
        logger.info("=" * 50)
        logger.info("STARTING RESUME INGESTION")
        logger.info("=" * 50)

        # Ensure persistence directory exists and is clean.
        # If we reuse an existing Chroma directory, Chroma will keep/append documents,
        # which breaks validation and makes retrieval hard to reason about.
        if persist_dir.exists():
            try:
                has_contents = any(persist_dir.iterdir())
            except Exception:
                has_contents = True
            if has_contents:
                logger.warning(f"Persistence directory not empty; resetting: {persist_dir}")
                shutil.rmtree(persist_dir)

        persist_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Persistence directory ready: {persist_dir}")

        # Stage 1: Load and validate resume document
        load_start = time.time()
        raw_docs = load_resume_documents(resume_path)
        ingestion_stats["load_time"] = time.time() - load_start

        # Stage 2: Intelligent document chunking
        chunk_start = time.time()
        chunks = chunk_documents(raw_docs)  # Resume-aware chunking
        chunks = validate_chunks(chunks)    # Quality validation
        ingestion_stats["chunk_time"] = time.time() - chunk_start
        ingestion_stats["chunks_created"] = len(chunks)

        # Stage 3: Initialize embedding model
        embed_start = time.time()
        embeddings = build_embeddings()
        ingestion_stats["embedding_time"] = time.time() - embed_start

        # Stage 4: Prepare chunks for vector store (filter complex metadata)
        filtered_chunks = filter_complex_metadata(chunks)
        logger.info(f"Filtered metadata for {len(filtered_chunks)} chunks")

        # Stage 5: Create and persist vector store
        store_start = time.time()
        client_settings = Settings(anonymized_telemetry=False)
        vector_store = Chroma.from_documents(
            documents=filtered_chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
            client_settings=client_settings,
        )
        ingestion_stats["store_time"] = time.time() - store_start

        # Stage 6: Validate the created vector store
        if not validate_vector_store(persist_dir, len(filtered_chunks)):
            raise RuntimeError("Vector store validation failed")

        # Calculate final performance statistics
        ingestion_stats["total_time"] = time.time() - ingestion_stats["start_time"]
        ingestion_stats["success"] = True

        logger.info("=" * 50)
        logger.info("INGESTION COMPLETED SUCCESSFULLY")
        logger.info(f"Total chunks: {len(filtered_chunks)}")
        logger.info(f"Total time: {ingestion_stats['total_time']:.2f}s")
        logger.info(f"Avg chunk size: {sum(len(c.page_content) for c in filtered_chunks) / len(filtered_chunks):.0f} chars")
        logger.info("=" * 50)

        return ingestion_stats

    except Exception as e:
        # Capture error information for debugging
        ingestion_stats["total_time"] = time.time() - ingestion_stats["start_time"]
        ingestion_stats["error"] = str(e)

        logger.error("=" * 50)
        logger.error("INGESTION FAILED")
        logger.error(f"Error: {e}")
        logger.error(f"Total time: {ingestion_stats['total_time']:.2f}s")
        logger.error("=" * 50)

        raise


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
        description="Ingest resume into Chroma vector store (production-ready).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --resume ./my_resume.pdf --persist-dir ./chroma_db
  python ingest.py --resume ./resume.pdf --persist-dir ./db --log-level DEBUG --log-file ./ingest.log
        """
    )
    parser.add_argument(
        "--resume",
        required=True,
        type=Path,
        help="Path to your master resume (PDF format).",
    )
    parser.add_argument(
        "--persist-dir",
        required=True,
        type=Path,
        help="Directory to store the Chroma vector database.",
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
        # Run ingestion with performance monitoring
        stats = ingest_resume(resume_path=args.resume, persist_dir=args.persist_dir)

        # Exit successfully
        exit(0)

    except KeyboardInterrupt:
        logger.warning("Ingestion interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Ingestion failed with error: {e}")
        exit(1)


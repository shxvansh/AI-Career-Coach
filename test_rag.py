#!/usr/bin/env python3
"""
RAG System Test Runner

Comprehensive testing suite for the RAG (Retrieval-Augmented Generation) system.
Tests the complete pipeline from resume ingestion to job matching queries.

Usage:
    python test_rag.py [--resume RESUME_FILE] [--jobs JOB_DIR] [--persist-dir DB_DIR]

Features:
- Automated end-to-end testing
- Performance benchmarking
- Quality validation metrics
- Comprehensive reporting
- Error handling and recovery
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import shutil

# Import RAG modules
from RAG.ingest import ingest_resume
from RAG.query import retrieve_resume_chunks

# For PDF generation from text
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("Warning: reportlab not available. Install with: pip install reportlab")

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    PdfReader = None  # type: ignore
    HAS_PYPDF = False


# Simple keyword expectations per sample job type (for reliability validation)
# This is intentionally lightweight: we want to validate that retrieval returns obviously relevant content.
EXPECTED_KEYWORDS = {
    "ai_ml_engineer": ["machine learning", "tensorflow", "pytorch", "nlp", "rag", "bert"],
    "data_scientist": ["data", "python", "sql", "model", "analysis", "statistics"],
    "frontend_dev": ["react", "javascript", "typescript", "css", "html"],
}


class RAGTester:
    """Comprehensive RAG system tester."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.test_results = {
            "timestamp": time.time(),
            "tests": [],
            "summary": {}
        }

    def read_pdf_text(self, pdf_file: Path) -> str:
        """Read PDF text using pypdf (best-effort)."""
        if not HAS_PYPDF:
            return ""
        try:
            reader = PdfReader(str(pdf_file))
            parts: List[str] = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                if txt.strip():
                    parts.append(txt)
            return "\n\n".join(parts)
        except Exception:
            return ""

    def convert_text_to_pdf(self, text_file: Path, pdf_file: Path) -> bool:
        """Convert text resume to PDF format."""
        if not HAS_REPORTLAB:
            print("Error: reportlab required for PDF conversion")
            return False

        try:
            # Read text content
            with open(text_file, 'r') as f:
                content = f.read()

            # Create PDF
            doc = SimpleDocTemplate(str(pdf_file), pagesize=letter)
            styles = getSampleStyleSheet()

            # Split content into paragraphs
            story = []
            for paragraph in content.split('\n\n'):
                if paragraph.strip():
                    p = Paragraph(paragraph.replace('\n', '<br/>'), styles['Normal'])
                    story.append(p)
                    story.append(Spacer(1, 12))

            doc.build(story)
            print(f"Converted {text_file} to {pdf_file}")
            return True

        except Exception as e:
            print(f"Error converting text to PDF: {e}")
            return False

    def run_ingestion_test(self, resume_path: Path) -> Dict[str, Any]:
        """Test resume ingestion pipeline."""
        print("=" * 50)
        print("TESTING RESUME INGESTION")
        print("=" * 50)

        start_time = time.time()
        success = False
        error = None
        stats = {}

        try:
            # Clean previous database if exists
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)

            # Run ingestion
            stats = ingest_resume(resume_path, self.persist_dir)
            success = stats.get('success', False)

            if success:
                print("✅ Ingestion completed successfully")
                print(f"   Chunks created: {stats.get('chunks_created', 0)}")
                ingestion_time = float(stats.get("total_time", time.time() - start_time))
                print(f"   Total time: {ingestion_time:.2f}s")
            else:
                print("❌ Ingestion failed")
                error = "Ingestion returned success=False"

        except Exception as e:
            error = str(e)
            print(f"❌ Ingestion failed with error: {e}")

        total_time = time.time() - start_time

        result = {
            "test_name": "resume_ingestion",
            "success": success,
            "total_time": total_time,
            "error": error,
            "stats": stats
        }

        self.test_results["tests"].append(result)
        return result

    def run_query_test(self, job_path: Path, test_name: str, job_text: Optional[str] = None) -> Dict[str, Any]:
        """Test job query against ingested resume."""
        print(f"\n{'=' * 50}")
        print(f"TESTING QUERY: {test_name.upper()}")
        print(f"{'=' * 50}")

        start_time = time.time()
        success = False
        error = None
        stats = {}
        chunks = []

        try:
            # Run query
            chunks, stats = retrieve_resume_chunks(
                job_pdf_path=job_path,
                persist_dir=self.persist_dir,
                k=6  # Test with 6 chunks
            )

            success = stats.get('success', False)

            if success:
                print("✅ Query completed successfully")
                print(f"   Retrieved chunks: {stats.get('total_chunks_retrieved', 0)}")
                print(f"   Returned chunks: {stats.get('chunks_returned', 0)}")
                query_time = float(stats.get("total_time", time.time() - start_time))
                print(f"   Total time: {query_time:.2f}s")
                # Show top relevance scores
                if chunks:
                    scores = [chunk.metadata.get('relevance_score', 0) for chunk in chunks]
                    avg_score = sum(scores) / len(scores) if scores else 0.0
                    print(f"   Avg relevance: {avg_score:.3f}")
                    print(f"   Relevance range: [{min(scores):.3f}, {max(scores):.3f}]")

                # Keyword coverage (reliability signal)
                # Prefer provided job_text (from .txt fixtures); fall back to PDF extraction if needed.
                effective_job_text = job_text or self.read_pdf_text(job_path)
                expected = EXPECTED_KEYWORDS.get(test_name, [])
                matched_keywords: List[str] = []
                if expected and effective_job_text:
                    haystack = "\n\n".join(c.page_content for c in chunks).lower() if chunks else ""
                    for kw in expected:
                        if kw.lower() in haystack:
                            matched_keywords.append(kw)
                    print(f"   Keyword hits: {len(matched_keywords)}/{len(expected)} ({', '.join(matched_keywords) if matched_keywords else 'none'})")
            else:
                print("❌ Query failed")
                error = "Query returned success=False"

        except Exception as e:
            error = str(e)
            print(f"❌ Query failed with error: {e}")

        total_time = time.time() - start_time

        result = {
            "test_name": f"query_{test_name}",
            "success": success,
            "total_time": total_time,
            "error": error,
            "stats": stats,
            "chunks_returned": len(chunks),
            "relevance_scores": [chunk.metadata.get('relevance_score', 0) for chunk in chunks] if chunks else [],
            "lexical_relevance_scores": [chunk.metadata.get('lexical_relevance_score', 0) for chunk in chunks] if chunks else [],
            "matched_keywords": matched_keywords if success else []
        }

        self.test_results["tests"].append(result)
        return result

    def validate_results(self) -> Dict[str, Any]:
        """Validate test results against expected criteria."""
        print(f"\n{'=' * 50}")
        print("VALIDATING RESULTS")
        print(f"{'=' * 50}")

        validation = {
            "overall_success": True,
            "checks": []
        }

        # Check ingestion success
        ingestion_test = next((t for t in self.test_results["tests"] if t["test_name"] == "resume_ingestion"), None)
        if not ingestion_test or not ingestion_test["success"]:
            validation["overall_success"] = False
            validation["checks"].append({
                "check": "ingestion_success",
                "passed": False,
                "message": "Resume ingestion failed"
            })
        else:
            validation["checks"].append({
                "check": "ingestion_success",
                "passed": True,
                "message": f"Ingestion completed in {ingestion_test['total_time']:.2f}s"
            })

        # Check query tests
        query_tests = [t for t in self.test_results["tests"] if t["test_name"].startswith("query_")]

        for test in query_tests:
            test_name = test["test_name"].replace("query_", "")

            # Check query success
            if not test["success"]:
                validation["overall_success"] = False
                validation["checks"].append({
                    "check": f"{test_name}_query_success",
                    "passed": False,
                    "message": f"{test_name} query failed"
                })
                continue

            # Validate relevance scores
            scores = test.get("relevance_scores", [])
            if not scores:
                validation["checks"].append({
                    "check": f"{test_name}_relevance_scores",
                    "passed": False,
                    "message": f"{test_name}: No relevance scores generated"
                })
                validation["overall_success"] = False
                continue

            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)

            # With vector-score-based retrieval, absolute values vary by embedding/DB size.
            # We only enforce basic sanity: scores should be within [0,1] and avg should be non-trivial.
            score_valid = (0.0 <= min_score <= 1.0) and (0.0 <= max_score <= 1.0) and (avg_score >= 0.2)
            validation["checks"].append({
                "check": f"{test_name}_relevance_scores",
                "passed": score_valid,
                "message": f"Avg: {avg_score:.3f}, Range: [{min_score:.3f}, {max_score:.3f}]"
            })
            if not score_valid:
                validation["overall_success"] = False

            # Keyword coverage check (only for known fixture job types)
            expected_keywords = EXPECTED_KEYWORDS.get(test_name, [])
            matched = test.get("matched_keywords", []) or []
            if expected_keywords:
                keyword_pass = len(matched) >= max(1, len(expected_keywords) // 3)  # at least ~1/3 or 1
                validation["checks"].append({
                    "check": f"{test_name}_keyword_coverage",
                    "passed": keyword_pass,
                    "message": f"Matched {len(matched)}/{len(expected_keywords)} expected keywords"
                })
                if not keyword_pass:
                    validation["overall_success"] = False

            # Check chunk counts
            chunks_returned = test.get("chunks_returned", 0)
            chunk_count_valid = 1 <= chunks_returned <= 10  # Reasonable range

            validation["checks"].append({
                "check": f"{test_name}_chunk_count",
                "passed": chunk_count_valid,
                "message": f"{test_name}: {chunks_returned} chunks returned"
            })
            if not chunk_count_valid:
                validation["overall_success"] = False

        return validation

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate comprehensive test report."""
        # Calculate summary statistics
        total_tests = len(self.test_results["tests"])
        successful_tests = sum(1 for t in self.test_results["tests"] if t["success"])
        total_time = sum(t["total_time"] for t in self.test_results["tests"])

        self.test_results["summary"] = {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
            "total_time": total_time
        }

        # Generate report
        report = []
        report.append("=" * 60)
        report.append("RAG SYSTEM TEST REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.test_results['timestamp']))}")
        report.append("")

        report.append("SUMMARY:")
        report.append(f"Total tests: {total_tests}")
        report.append(f"Successful tests: {successful_tests}")
        report.append(f"Success rate: {self.test_results['summary']['success_rate']:.1%}")
        report.append(f"Total time: {total_time:.2f}s")
        report.append("")

        report.append("DETAILED RESULTS:")
        for test in self.test_results["tests"]:
            status = "✅ PASS" if test["success"] else "❌ FAIL"
            report.append(f"\n{status} {test['test_name'].upper()}")
            report.append(f"   Time: {test.get('total_time', 0.0):.2f}s")
            if test.get("stats"):
                if "chunks_created" in test["stats"]:
                    report.append(f"   Chunks created: {test['stats']['chunks_created']}")
                if "chunks_returned" in test.get("stats", {}):
                    report.append(f"   Chunks returned: {test['stats']['chunks_returned']}")

            if test["error"]:
                report.append(f"   Error: {test['error']}")

        # Validation results
        validation = self.validate_results()
        report.append("\nVALIDATION CHECKS:")
        for check in validation["checks"]:
            status = "✅ PASS" if check["passed"] else "❌ FAIL"
            report.append(f"   {status} {check['check']}: {check['message']}")

        overall_status = "✅ ALL TESTS PASSED" if validation["overall_success"] else "❌ SOME TESTS FAILED"
        report.append(f"\n{overall_status}")

        final_report = "\n".join(report)

        # Save to file if requested
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(final_report)
                f.write("\n\nRAW RESULTS:\n")
                json.dump(self.test_results, f, indent=2, default=str)
            print(f"\nReport saved to: {output_file}")

        return final_report


def main():
    parser = argparse.ArgumentParser(description="Test RAG system with sample data")
    parser.add_argument(
        "--resume",
        type=Path,
        default=Path("test_data/resume/sample_resume.txt"),
        help="Path to resume file (text or PDF)"
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=Path("test_data/job_descriptions"),
        help="Directory containing job description files"
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("test_data/test_db"),
        help="Directory for test vector database"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_results.txt"),
        help="Output file for test report"
    )
    parser.add_argument(
        "--skip-pdf-conversion",
        action="store_true",
        help="Skip PDF conversion (assume resume is already PDF)"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = RAGTester(args.persist_dir)

    # Convert resume to PDF if needed
    resume_pdf = args.resume
    if args.skip_pdf_conversion and args.resume.suffix.lower() != ".pdf":
        print("Error: --skip-pdf-conversion was set but resume is not a PDF.")
        print(f"  Got: {args.resume}")
        return 1

    if args.resume.suffix.lower() != ".pdf" and not args.skip_pdf_conversion:
        resume_pdf = args.resume.parent / f"{args.resume.stem}.pdf"
        if not tester.convert_text_to_pdf(args.resume, resume_pdf):
            print("Failed to convert resume to PDF. Install reportlab: pip install reportlab")
            return 1

    # Run ingestion test
    ingestion_result = tester.run_ingestion_test(resume_pdf)
    if not ingestion_result["success"]:
        print("Ingestion failed, skipping query tests")
        report = tester.generate_report(args.output)
        print(report)
        return 1

    # Run query tests for all job files
    if args.jobs_dir.exists():
        if args.skip_pdf_conversion:
            job_files = sorted(args.jobs_dir.glob("*.pdf"))
            if not job_files:
                print(f"Error: --skip-pdf-conversion was set but no .pdf job files found in {args.jobs_dir}")
                return 1
            for job_pdf in job_files:
                test_name = job_pdf.stem.replace("_job", "").replace("_", "_")
                tester.run_query_test(job_pdf, test_name)
        else:
            for job_file in sorted(args.jobs_dir.glob("*.txt")):
                # Convert job description to PDF
                job_pdf = job_file.parent / f"{job_file.stem}.pdf"
                if not tester.convert_text_to_pdf(job_file, job_pdf):
                    print(f"Skipping {job_file.name} - PDF conversion failed")
                    continue
                try:
                    job_text = job_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    job_text = ""
                test_name = job_file.stem.replace("_job", "").replace("_", "_")
                tester.run_query_test(job_pdf, test_name, job_text=job_text)
    else:
        print(f"Warning: Jobs directory {args.jobs_dir} not found")

    # Generate and display report
    report = tester.generate_report(args.output)
    print(report)

    # Return appropriate exit code
    validation = tester.validate_results()
    return 0 if validation["overall_success"] else 1


if __name__ == "__main__":
    exit(main())
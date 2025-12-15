#!/usr/bin/env python3
"""
Manual RAG Testing Interface

Interactive testing tool for exploring and validating the RAG system.
Allows users to:
- Test with custom job descriptions
- Explore retrieved chunks in detail
- Compare different queries
- Debug retrieval quality
- Understand relevance scoring

Usage:
    python manual_test_rag.py --persist-dir /path/to/db

Features:
- Interactive query testing
- Detailed chunk inspection
- Performance analysis
- Result comparison
- Export capabilities
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import textwrap

from RAG.query import retrieve_resume_chunks_from_text, load_vector_store, validate_retrieval_params


class ManualRAGTester:
    """Interactive RAG testing interface."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.vector_store = None
        self.query_history = []

        # Load vector store
        try:
            print(f"Loading vector store from: {persist_dir}")
            self.vector_store = load_vector_store(persist_dir)
            print("✅ Vector store loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load vector store: {e}")
            print("Make sure to run ingestion first: python RAG/ingest.py --resume <resume.pdf> --persist-dir <db_dir>")
            return

    def test_query(self, job_text: str, k: int = 6) -> Dict[str, Any]:
        """Test a single query and return detailed results."""
        print(f"\n{'='*60}")
        print(f"TESTING QUERY (k={k})")
        print(f"{'='*60}")

        # Validate parameters
        k = validate_retrieval_params(k)

        # Run query directly from text (no PDF required)
        start_time = time.time()
        chunks, stats = retrieve_resume_chunks_from_text(
            job_text=job_text,
            persist_dir=self.persist_dir,
            k=k,
            job_source="manual_input",
        )
        query_time = time.time() - start_time

        # Store in history
        result = {
            "timestamp": time.time(),
            "query_text": job_text[:100] + "..." if len(job_text) > 100 else job_text,
            "k": k,
            "chunks": chunks,
            "stats": stats,
            "query_time": query_time
        }
        self.query_history.append(result)

        return result

    def display_results(self, result: Dict[str, Any], show_chunks: bool = True):
        """Display query results in a user-friendly format."""
        stats = result["stats"]

        print(f"\n📊 PERFORMANCE STATS:")
        print(f"   Total time: {result['query_time']:.2f}s")
        print(f"   Retrieved chunks: {stats.get('total_chunks_retrieved', 0)}")
        print(f"   Returned chunks: {stats.get('chunks_returned', 0)}")
        print(f"   Job extraction: {stats.get('job_extraction_time', 0):.2f}s")
        print(f"   Vector search: {stats.get('vector_search_time', 0):.2f}s")
        print(f"   Validation: {stats.get('validation_time', 0):.2f}s")

        if not result["chunks"]:
            print("\n❌ No chunks returned")
            return

        # Show relevance statistics
        scores = [chunk.metadata.get('relevance_score', 0) for chunk in result["chunks"]]
        if scores:
            print(f"\n🎯 RELEVANCE SCORES:")
            print(f"   Avg: {sum(scores)/len(scores):.3f}")
            print(f"   Min: {min(scores):.3f}")
            print(f"   Max: {max(scores):.3f}")
            # Show score distribution
            score_ranges = {"high": 0, "medium": 0, "low": 0}
            for score in scores:
                if score >= 0.5:
                    score_ranges["high"] += 1
                elif score >= 0.2:
                    score_ranges["medium"] += 1
                else:
                    score_ranges["low"] += 1

            print(f"   Distribution - High: {score_ranges['high']}, Medium: {score_ranges['medium']}, Low: {score_ranges['low']}")

        if show_chunks:
            self.display_chunks(result["chunks"])

    def display_chunks(self, chunks: List[Any]):
        """Display chunks with detailed metadata."""
        print(f"\n📄 RETRIEVED CHUNKS ({len(chunks)}):")

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata

            # Header with key info
            section = metadata.get('section_type', 'unknown')
            relevance = metadata.get('relevance_score', 0.0)
            chunk_size = len(chunk.page_content)

            print(f"\n--- CHUNK {i} ---")
            print(f"Section: {section} | Relevance: {relevance:.3f} | Size: {chunk_size} chars")
            print(f"Chunk Index: {metadata.get('chunk_index', 'N/A')} | Total in Section: {metadata.get('total_chunks_in_section', 'N/A')}")

            # Content preview (truncated)
            content = chunk.page_content.strip()
            preview_length = 400

            if len(content) <= preview_length:
                print(f"\n{content}")
            else:
                print(f"\n{content[:preview_length]}")
                print(f"... [truncated, {len(content) - preview_length} more chars]")

            print("-" * 60)

    def interactive_mode(self):
        """Run interactive testing mode."""
        print("\n🤖 RAG Manual Testing Interface")
        print("=" * 40)
        print("Commands:")
        print("  'query' or 'q' - Test a query")
        print("  'file' or 'f' - Test with job description file")
        print("  'history' or 'h' - Show query history")
        print("  'compare' or 'c' - Compare last two queries")
        print("  'export' or 'e' - Export results to file")
        print("  'help' - Show this help")
        print("  'quit' or 'exit' - Exit")
        print()

        while True:
            try:
                cmd = input("rag-test> ").strip().lower()

                if cmd in ['quit', 'exit', 'q']:
                    break
                elif cmd in ['help', 'h', '?']:
                    self.interactive_mode()
                    return
                elif cmd in ['query', 'q']:
                    self._handle_query_command()
                elif cmd in ['file', 'f']:
                    self._handle_file_command()
                elif cmd in ['history', 'h']:
                    self._show_history()
                elif cmd in ['compare', 'c']:
                    self._compare_queries()
                elif cmd in ['export', 'e']:
                    self._export_results()
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def _handle_query_command(self):
        """Handle interactive query input."""
        print("\nEnter job description text (press Ctrl+D when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass

        job_text = '\n'.join(lines).strip()
        if not job_text:
            print("No text entered")
            return

        k = input("Number of chunks to retrieve (default 6): ").strip()
        k = int(k) if k.isdigit() else 6

        result = self.test_query(job_text, k)
        self.display_results(result)

    def _handle_file_command(self):
        """Handle file-based query testing."""
        file_path = input("Enter path to job description file: ").strip()
        if not file_path:
            return

        file_path = Path(file_path)
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return

        try:
            with open(file_path, 'r') as f:
                job_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return

        k = input("Number of chunks to retrieve (default 6): ").strip()
        k = int(k) if k.isdigit() else 6

        result = self.test_query(job_text, k)
        self.display_results(result)

    def _show_history(self):
        """Show query history."""
        if not self.query_history:
            print("No queries in history")
            return

        print(f"\n📚 QUERY HISTORY ({len(self.query_history)} queries):")
        for i, query in enumerate(self.query_history, 1):
            qt = float(query.get("query_time", 0.0))
            print(f"{i}. [{time.strftime('%H:%M:%S', time.localtime(query['timestamp']))}] "
                  f"k={query['k']}, chunks={len(query['chunks'])}, "
                  f"time={qt:.2f}s, query: {query['query_text']}")

    def _compare_queries(self):
        """Compare the last two queries."""
        if len(self.query_history) < 2:
            print("Need at least 2 queries to compare")
            return

        q1, q2 = self.query_history[-2], self.query_history[-1]

        print(f"\n🔍 COMPARING LAST TWO QUERIES:")
        print(f"Query 1: {q1['query_text']}")
        print(f"Query 2: {q2['query_text']}")
        print()

        # Compare stats
        print("PERFORMANCE COMPARISON:")
        for stat in ['total_chunks_retrieved', 'chunks_returned', 'vector_search_time', 'validation_time']:
            v1 = q1['stats'].get(stat, 0)
            v2 = q2['stats'].get(stat, 0)
            diff = v2 - v1
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            print(f"  {stat}: {v1:.2f} → {v2:.2f} {symbol}{abs(diff):.2f}")

        # Compare relevance scores
        scores1 = [c.metadata.get('relevance_score', 0) for c in q1['chunks']]
        scores2 = [c.metadata.get('relevance_score', 0) for c in q2['chunks']]

        if scores1 and scores2:
            avg1 = sum(scores1) / len(scores1)
            avg2 = sum(scores2) / len(scores2)

            print("\nRELEVANCE COMPARISON:")
            print(f"  Avg relevance (Query 1): {avg1:.3f}")
            print(f"  Avg relevance (Query 2): {avg2:.3f}")
            diff = avg2 - avg1
            trend = "increased" if diff > 0 else "decreased" if diff < 0 else "unchanged"
            print(f"  Change: {diff:.3f} ({trend})")

    def _export_results(self):
        """Export query history to file."""
        if not self.query_history:
            print("No results to export")
            return

        filename = input("Enter export filename (default: rag_test_results.json): ").strip()
        if not filename:
            filename = "rag_test_results.json"

        try:
            # Prepare export data
            export_data = {
                "export_timestamp": time.time(),
                "total_queries": len(self.query_history),
                "queries": []
            }

            for query in self.query_history:
                # Convert chunks to serializable format
                chunks_data = []
                for chunk in query["chunks"]:
                    chunk_data = {
                        "content": chunk.page_content,
                        "metadata": chunk.metadata
                    }
                    chunks_data.append(chunk_data)

                query_data = {
                    "timestamp": query["timestamp"],
                    "query_text": query["query_text"],
                    "k": query["k"],
                    "query_time": query["query_time"],
                    "stats": query["stats"],
                    "chunks": chunks_data
                }
                export_data["queries"].append(query_data)

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            print(f"Results exported to: {filename}")

        except Exception as e:
            print(f"Export failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Manual RAG testing interface")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("test_data/test_db"),
        help="Directory containing the vector database"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run single query and exit (instead of interactive mode)"
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Run query from file and exit"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=6,
        help="Number of chunks to retrieve"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = ManualRAGTester(args.persist_dir)
    if not tester.vector_store:
        return 1

    # Handle single query mode
    if args.query or args.query_file:
        if args.query_file:
            try:
                with open(args.query_file, 'r') as f:
                    job_text = f.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                return 1
        else:
            job_text = args.query

        result = tester.test_query(job_text, args.k)
        tester.display_results(result)
        return 0

    # Interactive mode
    tester.interactive_mode()
    return 0


if __name__ == "__main__":
    exit(main())
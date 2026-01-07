#!/usr/bin/env python3
"""
RAG Evaluation Metrics

Comprehensive evaluation framework for RAG system quality assessment.
Provides quantitative metrics for retrieval quality, relevance, and performance.

Usage:
    from rag_evaluation import RAGEvaluator
    evaluator = RAGEvaluator()
    metrics = evaluator.evaluate_query_results(query_results, ground_truth)
"""

import re
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import numpy as np


class RAGEvaluator:
    """Comprehensive RAG system evaluator."""

    def __init__(self):
        self.section_keywords = {
            'experience': {'experience', 'work', 'employment', 'professional', 'career'},
            'skills': {'skills', 'technologies', 'technical', 'competencies', 'expertise'},
            'education': {'education', 'academic', 'degree', 'university', 'college'},
            'projects': {'projects', 'portfolio', 'achievements', 'accomplishments'},
            'summary': {'summary', 'objective', 'profile', 'overview'},
            'contact': {'contact', 'personal', 'information', 'details'}
        }

    def evaluate_query_results(self, chunks: List[Any], job_description: str,
                             ground_truth: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of RAG query results.

        Args:
            chunks: Retrieved document chunks
            job_description: Original job description text
            ground_truth: Expected results (optional)

        Returns:
            Dictionary with various evaluation metrics
        """
        if not chunks:
            return self._empty_results_metrics()

        metrics = {
            'basic_stats': self._calculate_basic_stats(chunks),
            'relevance_metrics': self._calculate_relevance_metrics(chunks, job_description),
            'quality_metrics': self._calculate_quality_metrics(chunks),
            'section_distribution': self._analyze_section_distribution(chunks),
            'content_analysis': self._analyze_content_quality(chunks, job_description)
        }

        if ground_truth:
            metrics['ground_truth_comparison'] = self._compare_with_ground_truth(chunks, ground_truth)

        return metrics

    def _calculate_basic_stats(self, chunks: List[Any]) -> Dict[str, Any]:
        """Calculate basic statistical metrics."""
        if not chunks:
            return {'total_chunks': 0}

        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        relevance_scores = [chunk.metadata.get('relevance_score', 0) for chunk in chunks]

        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': np.mean(chunk_lengths),
            'median_chunk_length': np.median(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'avg_relevance_score': np.mean(relevance_scores),
            'median_relevance_score': np.median(relevance_scores),
            'relevance_score_std': np.std(relevance_scores),
            'high_relevance_chunks': sum(1 for s in relevance_scores if s >= 0.5),
            'medium_relevance_chunks': sum(1 for s in relevance_scores if 0.2 <= s < 0.5),
            'low_relevance_chunks': sum(1 for s in relevance_scores if s < 0.2)
        }

    def _calculate_relevance_metrics(self, chunks: List[Any], job_description: str) -> Dict[str, Any]:
        """Calculate relevance-based metrics."""
        if not chunks or not job_description:
            return {}

        job_words = set(re.findall(r'\b\w+\b', job_description.lower()))
        total_job_words = len(job_words)

        chunk_relevances = []
        keyword_matches = []

        for chunk in chunks:
            chunk_text = chunk.page_content.lower()
            chunk_words = set(re.findall(r'\b\w+\b', chunk_text))

            # Jaccard similarity
            intersection = len(job_words.intersection(chunk_words))
            union = len(job_words.union(chunk_words))
            jaccard = intersection / union if union > 0 else 0

            # Keyword coverage
            keyword_coverage = intersection / total_job_words if total_job_words > 0 else 0

            chunk_relevances.append(jaccard)
            keyword_matches.append(keyword_coverage)

        return {
            'avg_jaccard_similarity': np.mean(chunk_relevances),
            'median_jaccard_similarity': np.median(chunk_relevances),
            'avg_keyword_coverage': np.mean(keyword_matches),
            'median_keyword_coverage': np.median(keyword_matches),
            'relevance_consistency': 1 - np.std(chunk_relevances),  # Lower std = more consistent
            'top_chunk_relevance': max(chunk_relevances) if chunk_relevances else 0
        }

    def _calculate_quality_metrics(self, chunks: List[Any]) -> Dict[str, Any]:
        """Calculate content quality metrics."""
        if not chunks:
            return {}

        quality_scores = []

        for chunk in chunks:
            content = chunk.page_content
            score = 0

            # Length appropriateness (chunks shouldn't be too short or long)
            length = len(content)
            if 100 <= length <= 1000:
                score += 0.3
            elif 50 <= length <= 1500:
                score += 0.2

            # Content diversity (not just whitespace/special chars)
            alpha_chars = len(re.findall(r'[a-zA-Z]', content))
            total_chars = len(content)
            if total_chars > 0:
                alpha_ratio = alpha_chars / total_chars
                if alpha_ratio > 0.7:
                    score += 0.3
                elif alpha_ratio > 0.5:
                    score += 0.2

            # Structure bonus (has some formatting)
            if '\n' in content or '.' in content:
                score += 0.2

            # Metadata completeness
            metadata = chunk.metadata
            metadata_score = sum([
                0.1 for key in ['section_type', 'chunk_index', 'relevance_score']
                if key in metadata and metadata[key] is not None
            ])
            score += metadata_score

            quality_scores.append(min(score, 1.0))  # Cap at 1.0

        return {
            'avg_quality_score': np.mean(quality_scores),
            'median_quality_score': np.median(quality_scores),
            'quality_consistency': 1 - np.std(quality_scores),
            'high_quality_chunks': sum(1 for s in quality_scores if s >= 0.7),
            'low_quality_chunks': sum(1 for s in quality_scores if s < 0.4)
        }

    def _analyze_section_distribution(self, chunks: List[Any]) -> Dict[str, Any]:
        """Analyze distribution of chunks across resume sections."""
        if not chunks:
            return {}

        section_counts = defaultdict(int)
        total_chunks = len(chunks)

        for chunk in chunks:
            section = chunk.metadata.get('section_type', 'unknown')
            section_counts[section] += 1

        # Calculate percentages
        section_distribution = {
            section: {
                'count': count,
                'percentage': count / total_chunks
            }
            for section, count in section_counts.items()
        }

        # Identify dominant sections
        dominant_sections = sorted(
            [(s, c) for s, c in section_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]

        return {
            'section_distribution': section_distribution,
            'dominant_sections': dominant_sections,
            'section_diversity': len(section_counts),  # Number of different sections
            'most_common_section': dominant_sections[0][0] if dominant_sections else None
        }

    def _analyze_content_quality(self, chunks: List[Any], job_description: str) -> Dict[str, Any]:
        """Analyze content quality and relevance patterns."""
        if not chunks:
            return {}

        # Extract job requirements
        job_lower = job_description.lower()
        job_requirements = set()

        # Look for common requirement patterns
        patterns = [
            r'required:?\s*(.*?)(?:\n|$)',
            r'experience.*?(?:with|in)\s*(.*?)(?:\n|$)',
            r'skills?:?\s*(.*?)(?:\n|$)',
            r'knowledge.*?(?:of|in)\s*(.*?)(?:\n|$)',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, job_lower, re.IGNORECASE)
            for match in matches:
                # Extract individual skills/requirements
                items = re.split(r'[,&]|\sand\s|\sor\s', match)
                job_requirements.update(item.strip().lower() for item in items if len(item.strip()) > 2)

        # Analyze chunk content for requirement matches
        requirement_matches = []
        technical_matches = []

        for chunk in chunks:
            content_lower = chunk.page_content.lower()
            chunk_matches = 0
            tech_matches = 0

            # Count requirement matches
            for req in job_requirements:
                if req in content_lower:
                    chunk_matches += 1

            # Count technical skill mentions (simple heuristic)
            tech_keywords = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes',
                           'tensorflow', 'pytorch', 'react', 'node', 'api', 'database', 'ml', 'ai']
            for tech in tech_keywords:
                if tech in content_lower:
                    tech_matches += 1

            requirement_matches.append(chunk_matches)
            technical_matches.append(tech_matches)

        return {
            'avg_requirement_matches': np.mean(requirement_matches),
            'total_requirement_matches': sum(requirement_matches),
            'avg_technical_matches': np.mean(technical_matches),
            'chunks_with_requirements': sum(1 for m in requirement_matches if m > 0),
            'chunks_with_tech_skills': sum(1 for m in technical_matches if m > 0),
            'requirement_coverage': len([m for m in requirement_matches if m > 0]) / len(chunks)
        }

    def _compare_with_ground_truth(self, chunks: List[Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results with ground truth expectations."""
        # This would be customized based on your ground truth format
        # For now, return a placeholder structure
        return {
            'ground_truth_available': True,
            'comparison_metrics': {
                'relevance_accuracy': None,  # Would calculate precision/recall against ground truth
                'section_accuracy': None,    # Would compare predicted vs actual sections
            }
        }

    def _empty_results_metrics(self) -> Dict[str, Any]:
        """Return metrics for empty results."""
        return {
            'basic_stats': {'total_chunks': 0},
            'relevance_metrics': {},
            'quality_metrics': {},
            'section_distribution': {},
            'content_analysis': {},
            'error': 'No chunks returned'
        }

    def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        if 'error' in evaluation_results:
            return f"❌ Evaluation Error: {evaluation_results['error']}"

        report = []
        report.append("📊 RAG Evaluation Report")
        report.append("=" * 50)

        # Basic stats
        basic = evaluation_results.get('basic_stats', {})
        if basic:
            report.append("\n🔢 Basic Statistics:")
            report.append(f"   Total chunks: {basic.get('total_chunks', 0)}")
            report.append(f"   Avg relevance: {basic.get('avg_relevance_score', 0):.3f}")
            report.append(f"   Chunk length: {basic.get('avg_chunk_length', 0):.0f} chars avg")
            report.append(f"   High relevance: {basic.get('high_relevance_chunks', 0)} chunks")

        # Quality metrics
        quality = evaluation_results.get('quality_metrics', {})
        if quality:
            report.append("\n⭐ Quality Metrics:")
            report.append(f"   Avg quality score: {quality.get('avg_quality_score', 0):.3f}")
            report.append(f"   High quality chunks: {quality.get('high_quality_chunks', 0)}")

        # Section distribution
        sections = evaluation_results.get('section_distribution', {})
        dist = sections.get('section_distribution', {})
        if dist:
            report.append("\n📑 Section Distribution:")
            for section, data in sorted(dist.items(), key=lambda x: x[1]['count'], reverse=True):
                report.append(f"   {section}: {data['count']} chunks ({data['percentage']:.1%})")

        # Content analysis
        content = evaluation_results.get('content_analysis', {})
        if content:
            report.append("\n🎯 Content Analysis:")
            report.append(f"   Chunks with requirements: {content.get('chunks_with_requirements', 0)}")
            report.append(f"   Avg requirement matches: {content.get('avg_requirement_matches', 0):.2f}")

        return "\n".join(report)

    def evaluate_test_suite(self, test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate an entire test suite."""
        if not test_results:
            return {'error': 'No test results provided'}

        suite_metrics = {
            'total_tests': len(test_results),
            'successful_tests': sum(1 for t in test_results if t.get('success', False)),
            'total_time': sum(t.get('total_time', 0) for t in test_results),
            'avg_relevance_scores': [],
            'performance_metrics': []
        }

        for test in test_results:
            if test.get('success') and 'chunks' in test:
                chunks = test['chunks']
                if chunks:
                    scores = [c.metadata.get('relevance_score', 0) for c in chunks]
                    suite_metrics['avg_relevance_scores'].append(np.mean(scores))

            if 'stats' in test:
                suite_metrics['performance_metrics'].append({
                    'query_time': test.get('total_time', 0),
                    'chunks_returned': test['stats'].get('chunks_returned', 0)
                })

        # Calculate aggregates
        if suite_metrics['avg_relevance_scores']:
            suite_metrics['overall_avg_relevance'] = np.mean(suite_metrics['avg_relevance_scores'])
            suite_metrics['relevance_consistency'] = 1 - np.std(suite_metrics['avg_relevance_scores'])

        suite_metrics['success_rate'] = suite_metrics['successful_tests'] / suite_metrics['total_tests']

        return suite_metrics


# Convenience functions for easy usage
def evaluate_single_query(chunks: List[Any], job_description: str) -> Dict[str, Any]:
    """Quick evaluation of a single query result."""
    evaluator = RAGEvaluator()
    return evaluator.evaluate_query_results(chunks, job_description)


def generate_evaluation_report(results: Dict[str, Any]) -> str:
    """Generate a formatted evaluation report."""
    evaluator = RAGEvaluator()
    return evaluator.generate_report(results)


if __name__ == "__main__":
    # Example usage
    print("RAG Evaluation Module")
    print("Import and use RAGEvaluator class for comprehensive evaluation")
    print("See docstrings for usage examples")
#!/bin/bash

# Quick RAG Testing Script
# Runs a complete test cycle to verify RAG system functionality

echo "🚀 Starting RAG System Quick Test"
echo "=================================="

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create test database directory
mkdir -p test_data/test_db

echo "📝 Step 1: Converting resume to PDF..."
if python test_rag.py --resume test_data/resume/sample_resume.txt --persist-dir test_data/test_db --skip-pdf-conversion --output /tmp/quick_test.txt 2>/dev/null; then
    echo "✅ Resume ingestion test passed"
else
    echo "❌ Resume ingestion test failed"
    exit 1
fi

echo "🔍 Step 2: Testing query functionality..."
if python manual_test_rag.py --persist-dir test_data/test_db --query "Python machine learning developer with TensorFlow experience" --k 3 >/dev/null 2>&1; then
    echo "✅ Query test passed"
else
    echo "❌ Query test failed"
    exit 1
fi

echo "📊 Step 3: Running evaluation..."
if python -c "
from rag_evaluation import evaluate_single_query, generate_evaluation_report
from RAG.query import retrieve_resume_chunks
import json

# Run a test query
chunks, stats = retrieve_resume_chunks('test_data/job_descriptions/ai_ml_engineer_job.txt', 'test_data/test_db', 4)
if chunks:
    results = evaluate_single_query(chunks, 'AI ML engineer job')
    report = generate_evaluation_report(results)
    print('✅ Evaluation test passed')
    print('Sample metrics:')
    basic = results.get('basic_stats', {})
    print(f'  - Chunks: {basic.get(\"total_chunks\", 0)}')
    print(f'  - Avg relevance: {basic.get(\"avg_relevance_score\", 0):.3f}')
else:
    print('❌ No chunks returned')
    exit(1)
"; then
    echo "✅ Evaluation test passed"
else
    echo "❌ Evaluation test failed"
    exit 1
fi

echo ""
echo "🎉 All tests passed! RAG system is working correctly."
echo ""
echo "Next steps:"
echo "1. Run full test suite: python test_rag.py"
echo "2. Interactive testing: python manual_test_rag.py"
echo "3. Read testing guide: cat RAG_TESTING_README.md"
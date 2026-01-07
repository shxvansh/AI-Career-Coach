# RAG System Testing Guide

Complete testing suite for validating your Retrieval-Augmented Generation (RAG) system. This guide covers automated testing, manual validation, and quality assessment.

## 🏗️ Setup

### 1. Install Dependencies

Make sure you have the required packages:

```bash
pip install reportlab  # For PDF generation in tests
```

### 2. Prepare Test Data

The `test_data/` directory contains sample data for testing:

```
test_data/
├── resume/
│   └── sample_resume.txt          # Sample resume (converted to PDF automatically)
├── job_descriptions/
│   ├── ai_ml_engineer_job.txt     # High-relevance job (should match well)
│   ├── data_scientist_job.txt     # Medium-relevance job
│   └── frontend_dev_job.txt       # Low-relevance job (should match poorly)
└── expected_results/
    └── README.md                  # Expected test outcomes
```

## 🧪 Testing Methods

### Method 1: Automated Testing (`test_rag.py`)

Run comprehensive automated tests:

```bash
# Basic test with default settings
python test_rag.py

# Custom paths
python test_rag.py --resume /path/to/resume.pdf --jobs-dir /path/to/jobs --persist-dir /path/to/db

# Skip PDF conversion (if resume is already PDF)
python test_rag.py --skip-pdf-conversion

# Save detailed report
python test_rag.py --output test_report.txt
```

**What it tests:**
- ✅ Resume ingestion pipeline
- ✅ Job description queries (all 3 sample jobs)
- ✅ Performance timing
- ✅ Relevance scoring validation
- ✅ Result quality assessment

**Sample Output:**
```
============================================================
RAG SYSTEM TEST REPORT
============================================================
SUMMARY:
Total tests: 4
Successful tests: 4
Success rate: 100.0%
Total time: 45.23s

DETAILED RESULTS:

✅ PASS RESUME_INGESTION
   Total time: 12.45s
   Chunks created: 24

✅ PASS QUERY_AI_ML_ENGINEER
   Total time: 8.92s
   Chunks returned: 6
   Avg relevance: 0.623

✅ PASS QUERY_DATA_SCIENTIST
   Total time: 7.88s
   Chunks returned: 6
   Avg relevance: 0.345

✅ PASS QUERY_FRONTEND_DEV
   Total time: 6.78s
   Chunks returned: 3
   Avg relevance: 0.089

VALIDATION CHECKS:
   ✅ PASS ingestion_success: Ingestion completed in 12.45s
   ✅ PASS ai_ml_engineer_relevance_scores: Expected range [0.300, 0.800], Avg: 0.623, Range: [0.456, 0.789]
   ✅ PASS data_scientist_relevance_scores: Expected range [0.200, 0.600], Avg: 0.345, Range: [0.123, 0.567]
   ✅ PASS frontend_dev_relevance_scores: Expected range [0.000, 0.300], Avg: 0.089, Range: [0.034, 0.145]
```

### Method 2: Interactive Testing (`manual_test_rag.py`)

Explore the RAG system interactively:

```bash
# Start interactive mode
python manual_test_rag.py --persist-dir test_data/test_db

# Run single query from command line
python manual_test_rag.py --query "Looking for Python developer with ML experience"

# Run query from file
python manual_test_rag.py --query-file test_data/job_descriptions/ai_ml_engineer_job.txt
```

**Interactive Commands:**
```
rag-test> help
Commands:
  'query' or 'q' - Test a query
  'file' or 'f' - Test with job description file
  'history' or 'h' - Show query history
  'compare' or 'c' - Compare last two queries
  'export' or 'e' - Export results to file
  'quit' or 'exit' - Exit
```

**Sample Interactive Session:**
```
rag-test> query
Enter job description text (press Ctrl+D when done):
Senior Python developer with machine learning experience needed.
Experience with TensorFlow and data pipelines required.
Number of chunks to retrieve (default 6): 4

============================================================
TESTING QUERY (k=4)
============================================================

📊 PERFORMANCE STATS:
   Total time: 2.34s
   Retrieved chunks: 12
   Returned chunks: 4
   Job extraction: 0.12s
   Vector search: 1.89s
   Validation: 0.33s

🎯 RELEVANCE SCORES:
   Average: 0.634
   Highest: 0.789
   Lowest: 0.456
   Distribution - High: 3, Medium: 1, Low: 0
```

## 📊 Quality Metrics

### Relevance Scoring
- **High relevance (≥0.5)**: Strong match, should be included
- **Medium relevance (0.2-0.5)**: Moderate match, may be useful
- **Low relevance (<0.2)**: Weak match, likely noise

### Expected Results by Job Type

| Job Type | Expected Avg Relevance | Key Matching Content |
|----------|----------------------|---------------------|
| AI/ML Engineer | 0.3-0.8 | ML frameworks, RAG experience, cloud platforms |
| Data Scientist | 0.2-0.6 | Python, statistics, data pipelines, A/B testing |
| Frontend Dev | 0.0-0.3 | Minimal overlap (maybe JavaScript mention) |

### Performance Benchmarks

| Operation | Expected Time | Notes |
|-----------|---------------|-------|
| Resume Ingestion | <30s | Depends on resume length |
| Single Query | <5s | Includes text extraction + search |
| Full Test Suite | <60s | All tests combined |

## 🔍 Debugging & Validation

### Common Issues & Solutions

**1. Low Relevance Scores**
```
Problem: All queries return relevance < 0.2
Solution:
- Check if resume was properly ingested
- Verify embedding model consistency
- Ensure job descriptions contain relevant keywords
```

**2. No Chunks Returned**
```
Problem: Queries return empty results
Solution:
- Verify vector store was created successfully
- Check if persist directory exists and has content
- Ensure PDF text extraction worked
```

**3. Poor Section Classification**
```
Problem: Chunks have wrong section_type metadata
Solution:
- Check resume formatting (clear section headers)
- Verify section patterns in ingest.py
- Look at chunk content manually
```

### Manual Validation Steps

1. **Check Ingestion Quality:**
   ```bash
   python -c "
   from RAG.query import load_vector_store
   vs = load_vector_store('test_data/test_db')
   print(f'Documents in store: {vs._collection.count()}')
   "
   ```

2. **Inspect Chunk Content:**
   ```bash
   python manual_test_rag.py --persist-dir test_data/test_db
   # Then use 'query' command to examine results
   ```

3. **Compare Different Queries:**
   ```bash
   # In interactive mode
   rag-test> query
   # Enter: "Python machine learning developer"
   rag-test> query
   # Enter: "Frontend React developer"
   rag-test> compare
   ```

## 📈 Advanced Testing

### Custom Test Data

Create your own test cases:

1. **Add Resume:** Place in `test_data/resume/your_resume.pdf`
2. **Add Job:** Create `test_data/job_descriptions/your_job.txt`
3. **Update Expectations:** Modify `test_data/expected_results/README.md`

### Performance Testing

Test with larger datasets:

```bash
# Test with multiple resumes
python test_rag.py --resume large_resume.pdf

# Test batch queries
for job in test_data/job_descriptions/*.txt; do
    python manual_test_rag.py --query-file "$job" --k 8
done
```

### Integration Testing

Test the full pipeline:

```bash
# 1. Ingest resume
python RAG/ingest.py --resume test_data/resume/sample_resume.txt --persist-dir test_data/test_db

# 2. Test queries
python test_rag.py --persist-dir test_data/test_db

# 3. Interactive exploration
python manual_test_rag.py --persist-dir test_data/test_db
```

## 📋 Test Checklist

- [ ] Resume ingestion completes without errors
- [ ] Vector store contains expected number of chunks
- [ ] All job queries return results
- [ ] Relevance scores are in expected ranges
- [ ] Section types are correctly identified
- [ ] Chunk sizes are appropriate for content type
- [ ] Performance meets timing requirements
- [ ] Low-relevance jobs return fewer/no results
- [ ] High-relevance jobs return quality matches

## 🚀 Next Steps

Once testing passes:

1. **Integrate with Agent:** Connect RAG to your LangChain agent
2. **Add Fine-tuned Model:** Use your LoRA model for resume generation
3. **Web Interface:** Build FastAPI/Streamlit UI
4. **Production Deployment:** Set up proper infrastructure

The RAG system is now thoroughly tested and ready for integration! 🎉
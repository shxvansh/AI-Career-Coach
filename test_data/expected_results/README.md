# RAG Testing Expected Results

This directory contains expected outcomes for testing the RAG system with the sample data.

## Test Cases

### 1. AI/ML Engineer Job (High Relevance)
**Expected Behavior:**
- Should retrieve chunks from: experience, skills, projects sections
- High relevance scores (>0.5) for ML/AI related content
- Should identify relevant sections: experience, skills, projects
- Experience section chunks should have higher relevance than education

**Key Content to Match:**
- "Senior AI/ML Engineer" experience
- Technical skills: TensorFlow, PyTorch, Hugging Face, AWS
- RAG project experience
- NLP and ML pipeline work

### 2. Data Scientist Job (Medium Relevance)
**Expected Behavior:**
- Should retrieve some relevant chunks but lower scores than AI/ML job
- Should match: data science skills, Python, statistical modeling
- Lower relevance for pure ML/deep learning content
- Should still find some matching experience chunks

**Key Content to Match:**
- Python, ML frameworks, statistical analysis
- Data pipeline experience (Apache Airflow)
- A/B testing and experimentation
- SQL and data processing skills

### 3. Frontend Developer Job (Low Relevance)
**Expected Behavior:**
- Should retrieve very few or no highly relevant chunks
- Low relevance scores (<0.2) for most content
- May match basic programming skills but not specific technologies
- Should demonstrate system's ability to filter out irrelevant content

**Key Content (Limited Match):**
- Basic JavaScript knowledge (if mentioned)
- General programming experience
- No React, TypeScript, or frontend-specific skills

## Performance Expectations

### Timing:
- Ingestion: < 30 seconds for sample resume
- Query: < 5 seconds per job description
- Total pipeline: < 60 seconds

### Quality Metrics:
- AI/ML Job: > 70% of retrieved chunks should be relevant
- Data Scientist Job: > 40% of retrieved chunks should be relevant
- Frontend Job: < 20% of retrieved chunks should be relevant

### Chunk Distribution:
- Skills section: Smaller chunks (200-400 chars)
- Experience/Projects: Larger chunks (800-1200 chars)
- Education: Medium chunks (500-700 chars)

## Validation Checklist

- [ ] Resume ingestion completes without errors
- [ ] Vector store created successfully
- [ ] All three job queries return results
- [ ] Relevance scores are reasonable (0.0-1.0 range)
- [ ] Section types are correctly identified
- [ ] Chunk sizes are appropriate for section types
- [ ] Performance meets timing expectations
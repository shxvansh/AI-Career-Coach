# AI Career Coach: Resume Tuner & Gap Analyzer

This project is an AI-powered tool that helps you tailor your resume to specific job descriptions. It uses **Retrieval-Augmented Generation (RAG)** to find relevant experience from your master resume and a **Fine-Tuned LoRA Model (Qwen2.5-0.5B)** to act as a "Resume Expert," performing gap analysis and suggesting specific improvements.

## 🚀 Features

*   **RAG-based Retrieval**: Intelligently extracts the most relevant sections from your resume PDF based on the job description.
*   **LoRA Fine-Tuned Expert**: Uses a specialized 0.5B parameter model (Qwen2.5) fine-tuned on resume data to identify gaps and suggest rewrites.
*   **Resume Aware Chunking**: Splits resumes by section (Skills, Experience, Projects) for better retrieval context.
*   **Grounded Generation**: Ensures all suggestions are backed by evidence from your actual resume (no hallucinations).
*   **Modern Web UI**: Clean, responsive HTML/JS interface for easy drag-and-drop usage.
*   **Privacy First**: Runs locally on your machine (supports Apple Silicon MPS acceleration).

## 🛠️ Tech Stack

*   **LLM**: Qwen/Qwen2.5-0.5B-Instruct (Fine-tuned with LoRA)
*   **Orchestration**: LangChain, ChromaDB
*   **Backend**: FastAPI
*   **Frontend**: HTML5, CSS3, Vanilla JavaScript
*   **PDF Processing**: PyMuPDF (fitz)
*   **Training**: PEFT (LoRA), TRL, PyTorch

## 📂 Project Structure

```
/AI-Career-Coach
├── api/
│   └── main.py             # FastAPI backend (Ingest & Tune endpoints)
├── frontend/
│   ├── index.html          # Web Interface
│   ├── style.css           # Styling
│   └── app.js              # Frontend Logic
├── RAG/
│   ├── ingest.py           # Resume ingestion & chunking logic
│   └── query.py            # Retrieval & relevance scoring
├── finetuning/
│   ├── train.py            # LoRA training script
│   ├── inference.py        # Inference logic with guardrails
│   ├── dataset.jsonl       # Training data
│   └── resume-expert-lora/ # Trained adapter (generated)
├── resume_tuner.py         # CLI orchestrator (alternative to API)
└── run_api.sh              # Helper script to start the server
```

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install torch transformers peft datasets trl accelerate sentence-transformers bitsandbytes python-multipart uvicorn
```

### 2. Train the Model
This creates the "Resume Expert" adapter (~5-10 mins on M1 Mac).
```bash
python finetuning/train.py
```

### 3. Run the App
Starts the FastAPI backend and serves the frontend.
```bash
./run_api.sh
```

### 4. Use It
Open **http://localhost:8000** in your browser.
1.  **Upload Resume**: Ingests your resume into the local vector store.
2.  **Upload Job**: Analyzes the job description against your resume.
3.  **View Results**: See missing keywords, rewrite suggestions, and copy LaTeX bullets.

## 🧠 How It Works

1.  **Ingestion**: Your resume PDF is converted to text, split into semantic chunks (Skills, Experience, etc.), and stored in ChromaDB.
2.  **Retrieval**: When you upload a job description, the system finds the top-k (default 4) most relevant chunks from your resume using semantic search + relevance scoring.
3.  **Inference**: The fine-tuned Qwen model analyzes the retrieved chunks against the job description to find gaps and generate evidence-backed bullet points.
4.  **Guardrails**: The output is filtered to ensure it doesn't hallucinate new skills or technologies not found in your resume.

## 🧪 Testing

*   **Manual CLI Test**: `python manual_test_rag.py`
*   **Automated RAG Test**: `python test_rag.py`

# Project 1: The "Agentic Resume" (Autonomous Resume Tuner & Job Search Agent)

## 1. Project Overview

This project is a hybrid AI system that acts as an autonomous career agent. It addresses a core problem for job seekers: the need to manually tune a resume for every single job application.

This agent ingests a user's single "master" resume (as LaTeX) and a job description provided as a PDF. It extracts the job description, augments it with web-scraped requirements via Tavily search, performs a "gap analysis" using a specialized, finetuned model (the "Resume Expert") to identify missing keywords and skills, and automatically rewrites the resume. The agent outputs an ATS-friendly HTML file that you can convert to PDF, and it can find other similar jobs that are a strong match for the newly generated resume.

This system combines:
*   **Finetuning (LoRA):** To create a specialist "brain" for resume analysis.
*   **RAG:** To read and understand the contents of the user's base resume.
*   **Agentic Tools:** To read web pages, write files, compile code (LaTeX), and search the web.

## 2. Features

*   **Finetuned Gap Analysis:** Uses a custom LoRA-finetuned model to provide expert-level analysis of how a resume matches a job description.
*   **Automated Resume Generation:** Reads a `.tex` template, injects newly generated, tuned bullet points, and emits an ATS-friendly HTML output you can later convert to PDF.
*   **Web-Augmented Requirements:** Reads the job description from PDF and uses Tavily search to scrape the internet for additional job requirements and context.
*   **Smart Job Search:** Uses the newly tuned resume's qualifications to generate highly relevant search queries for other job postings.
*   **End-to-End Automation:** A "one-click" agent that turns a job PDF into a custom resume and a list of new leads.

## 3. System Architecture

This project is built in two phases: Finetuning (Phase 0) and The Agent (Phase 1).

### Phase 0: Finetuning the "Resume Expert" (LoRA)

1.  **Base Model:** A high-performing open-source model (e.g., CodeLlama-7B or Mistral-7B).
2.  **Dataset:** A custom, hand-made JSONL file with ~300 examples. Each example contains:
    *   `resume_context`: Text snippets from a resume.
    *   `job_description_context`: Text snippets from a job post.
    *   `output_analysis`: The "expert" analysis and new bullet points we want the model to learn to generate.
3.  **Training:** The model is finetuned using LoRA (via the Hugging Face `peft` library) to create a small, efficient "adapter" that teaches it the new skill of resume gap analysis.
4.  **Result:** A "Resume Expert" model that is specialized in generating context-aware, ATS-friendly resume content.

### Phase 1: The Agentic Workflow (LangChain)

This agent uses the "Resume Expert" model as its brain and is given a set of custom tools.

1.  **Input:** The user provides a PDF of a job description.
2.  **Tool 1: `job_pdf_reader`:** Extracts the full text of the job description from the PDF.
3.  **Tool 2: `tavily_requirements_scraper`:** Uses **Tavily Search API** to scrape the web for the role's requirements and related context, enriching the job signal.
4.  **Tool 3: `resume_retriever_tool` (RAG):** Queries the RAG pipeline (built from your original master resume) to get matching skills, experiences, and projects.
5.  **LLM Call (The "Brain"):** With `job_description_text`, `web_requirements`, and `resume_context`, the LoRA-fused "Resume Expert" performs gap analysis and generates new, targeted bullet points in LaTeX or HTML-ready text.
6.  **Tool 4: `html_resume_generator`:** Injects the new content into an ATS-friendly HTML template. (You can convert the HTML to PDF afterward.)
7.  **Tool 5: `job_search_tool`:** Generates smart search queries (e.g., "AI Engineer jobs with RAG and FastAPI") and retrieves 5-10 similar job URLs.
8.  **Output:** The agent provides an ATS-friendly HTML resume and a list of new, highly relevant job links.

## 4. Tech Stack

*   **Orchestration:** LangChain (Agents, LCEL)
*   **Model Finetuning:** Hugging Face transformers, peft (LoRA), datasets
*   **LLM (Brain):** Mistral-7B (or similar) + your custom LoRA adapter
*   **RAG Pipeline:**
    *   **Loader:** `TextLoader` / `PyMuPDFLoader` (to build the DB).
    *   **Embeddings:** `HuggingFaceEmbeddings` (e.g., `all-MiniLM-L6-v2`)
    *   **Vector Store:** `ChromaDB` (for persistent storage)
*   **Tools:**
*   **Tavily API:** For robust job search and advanced scraping of role requirements.
*   **PDF Text Extraction:** To read job description PDFs.
*   **WebBaseLoader:** For basic page text extraction.
*   **subprocess:** For executing LaTeX compilation if needed for legacy flows; HTML generation is the default output path.
*   **Serving (Optional):** FastAPI (for the agent API) & Streamlit (for the frontend).

## 5. Project Structure

```
/agentic-resume
|
├── /app
│   ├── main.py             # FastAPI app / Streamlit app
│   ├── agent.py            # Agent definition, tools, and logic
│   ├── tools.py            # Python code for `scrape_job_description`, `retrieve_resume_context`, etc.
│   └── build_rag.py        # Script to ingest master resume into ChromaDB
|
├── /finetuning
│   ├── train_lora.ipynb    # Jupyter notebook for training the LoRA model
│   ├── generate_data.py    # Script to generate synthetic training data
│   └── dataset.jsonl       # Your custom finetuning dataset
|
├── /resume_data
│   ├── my_master_resume.txt # Your "master" resume raw text
│   ├── template.tex         # The LaTeX template for final generation
│   └── /db                  # Persistent Chroma vector store
|
├── /output
│   └── tuned_resume.pdf    # The final PDF generated by the agent
|
└── README.md               # This documentation
```

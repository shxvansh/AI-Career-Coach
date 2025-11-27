import os
import subprocess
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Configuration ---
DB_PATH = "resume_data/db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEMPLATE_PATH = "resume_data/template.tex"
OUTPUT_DIR = "output"

# --- Tool 1: Job Scraper ---
def scrape_job_description(url: str) -> str:
    """
    Scrapes the text content from a given Job Description URL.
    Uses LangChain's WebBaseLoader.
    """
    try:
        print(f"Scraping URL: {url}...")
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        # distinct content from multiple docs if any, usually just one for a page
        content = "\n\n".join([d.page_content for d in docs])
        
        # Basic cleaning to remove excessive whitespace
        cleaned_content = " ".join(content.split())
        return cleaned_content
    except Exception as e:
        return f"Error scraping job description: {str(e)}"

# --- Tool 2: Resume Retriever (RAG) ---
def retrieve_resume_context(query: str, k: int = 4) -> List[str]:
    """
    Searches the Master Resume Vector DB for relevant skills/experiences
    matching the query (usually the job description text).
    """
    print(f"Retrieving resume context for query: {query[:50]}...")
    
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Load the existing vector store
    vector_store = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
    
    # Perform similarity search
    results = vector_store.similarity_search(query, k=k)
    
    # Return list of text chunks
    return [doc.page_content for doc in results]

# --- Tool 3: Job Search (Tavily) ---
# Note: This requires TAVILY_API_KEY in environment variables
from langchain_community.tools.tavily_search import TavilySearchResults

def search_similar_jobs(query: str) -> str:
    """
    Uses Tavily API to search for similar job postings.
    """
    try:
        tool = TavilySearchResults(max_results=5)
        results = tool.invoke(query)
        return str(results)
    except Exception as e:
        return f"Error searching for jobs: {str(e)}"

# --- Tool 4: LaTeX Compiler ---
def compile_resume_pdf(new_bullets: List[str]) -> str:
    """
    Compiles a new PDF resume by injecting the provided bullets into the LaTeX template.
    Returns the path to the generated PDF.
    """
    try:
        print("Compiling tailored resume PDF...")
        
        # 1. Read Template
        if not os.path.exists(TEMPLATE_PATH):
            return f"Error: Template not found at {TEMPLATE_PATH}"
            
        with open(TEMPLATE_PATH, "r") as f:
            template_content = f.read()
            
        # 2. Inject Bullets
        # Join list into a string of LaTeX items
        bullet_string = "\n".join(new_bullets) 
        # Or ensure they have \item prefix if not provided by LLM
        # bullet_string = "\n".join([f"\\item {b}" for b in new_bullets])
        
        filled_content = template_content.replace("%--AGENT_BULLETS_HERE--%", bullet_string)
        
        # 3. Write Temporary .tex File
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_tex_path = os.path.join(OUTPUT_DIR, "tuned_resume.tex")
        
        with open(output_tex_path, "w") as f:
            f.write(filled_content)
            
        # 4. Compile with pdflatex
        # Note: Requires pdflatex installed on system
        cmd = ["pdflatex", "-interaction=nonstopmode", "-output-directory", OUTPUT_DIR, output_tex_path]
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            # It often fails on first run or due to missing packages, but let's return log
            return f"LaTeX compilation failed. Log: {result.stdout.decode('utf-8', errors='ignore')}"
            
        pdf_path = os.path.join(OUTPUT_DIR, "tuned_resume.pdf")
        return f"Success! Resume generated at: {pdf_path}"
        
    except Exception as e:
        return f"Error compiling PDF: {str(e)}"

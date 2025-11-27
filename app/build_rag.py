import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Paths
RESUME_PATH = "resume_data/my_master_resume.txt"
DB_PATH = "resume_data/db"

def build_vector_store():
    print(f"Loading resume from {RESUME_PATH}...")
    if not os.path.exists(RESUME_PATH):
        print(f"Error: File not found at {RESUME_PATH}")
        return

    # 1. Load the resume text
    loader = TextLoader(RESUME_PATH)
    documents = loader.load()

    # 2. Split text into chunks
    # We use a small chunk size because resume sections are short and dense
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split resume into {len(chunks)} chunks.")

    # 3. Initialize Embeddings
    # 'all-MiniLM-L6-v2' is a fast, efficient model good for sentence similarity
    print("Initializing embeddings model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create and Persist Vector Store
    print(f"Creating Chroma vector store at {DB_PATH}...")
    # Chroma automatically persists when using a persistent_directory
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("RAG database built successfully!")

if __name__ == "__main__":
    # Ensure we have the necessary libraries installed
    # pip install langchain langchain-community langchain-huggingface langchain-chroma chromadb sentence-transformers
    build_vector_store()


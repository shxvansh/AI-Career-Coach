from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
from pathlib import Path
import json
from typing import Optional

# Import our internal modules
from RAG.ingest import ingest_resume
from RAG.query import load_job_pdf, retrieve_resume_chunks_from_text
from resume_tuner import _build_resume_context
from finetuning.inference import ResumeExpert

app = FastAPI(title="AI Career Coach API", version="1.0")

# CORS (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
PERSIST_DIR = Path("tuner_db")
ADAPTER_PATH = Path("finetuning/resume-expert-lora")

# Global model instance
expert_model: Optional[ResumeExpert] = None

def get_model():
    global expert_model
    if expert_model is None:
        if not ADAPTER_PATH.exists():
            print(f"WARNING: Adapter path {ADAPTER_PATH} not found. Running inference with base model only (or failing if strict).")
            # We can still run with base model if adapter is missing, or fail.
            # For now, let's assume base model is okay or let ResumeExpert handle it.
            # ResumeExpert(adapter_path=...) handles missing adapter by printing a warning?
            # Looking at inference.py: if adapter_path: ... else: self.model = self.base_model
            # So it is safe.
            pass
        expert_model = ResumeExpert(adapter_path=str(ADAPTER_PATH))
    return expert_model

@app.on_event("startup")
async def startup_event():
    # Pre-load model on startup to avoid delay on first request
    try:
        get_model()
    except Exception as e:
        print(f"Failed to load model on startup: {e}")

@app.post("/ingest")
async def ingest_resume_endpoint(file: UploadFile = File(...)):
    """
    Upload a resume PDF and ingest it into the RAG vector store.
    """
    try:
        file_path = UPLOAD_DIR / f"resume_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Ingest
        stats = ingest_resume(resume_path=file_path, persist_dir=PERSIST_DIR)
        
        return JSONResponse(content={
            "status": "success",
            "message": "Resume ingested successfully",
            "stats": stats,
            "filename": file.filename
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/tune")
async def tune_resume_endpoint(
    job_file: UploadFile = File(...), 
    k: int = 4
):
    """
    Upload a job description PDF, retrieve chunks from the previously ingested resume,
    and generate a resume gap analysis and tuning suggestions.
    """
    try:
        job_path = UPLOAD_DIR / f"job_{job_file.filename}"
        with open(job_path, "wb") as buffer:
            shutil.copyfileobj(job_file.file, buffer)

        # 1. Extract text from Job PDF
        job_docs = load_job_pdf(job_path)
        job_text = "\n\n".join(d.page_content for d in job_docs).strip()
        
        if not job_text:
            raise HTTPException(status_code=400, detail="Could not extract text from job PDF")

        # 2. Retrieve Resume Chunks (uses the global PERSIST_DIR)
        # Note: This assumes /ingest was called first to populate PERSIST_DIR
        chunks, retrieval_stats = retrieve_resume_chunks_from_text(
            job_text=job_text,
            persist_dir=PERSIST_DIR,
            k=k,
            job_source=str(job_path)
        )

        if not chunks:
             return JSONResponse(content={
                "status": "warning",
                "message": "No resume chunks found. Did you upload a resume first?",
                "analysis": None
            })

        # 3. Build Context
        resume_context = _build_resume_context(chunks)

        # 4. Generate Analysis
        model = get_model()
        model_output = model.generate_grounded_json(
            resume_context=resume_context, 
            job_description=job_text
        )

        # 5. Post-process / Guardrails (Reusing logic from resume_tuner.py)
        # We need to replicate the guardrail logic here or refactor resume_tuner to expose it.
        # Since I imported _mentions_unseen_tech etc, I can replicate it quickly.
        
        # Helper for tokenizing used in guardrails
        import re
        def _tokenize(s: str) -> set[str]:
             return {w for w in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(w) >= 3}
             
        # Re-creating the allowed text context
        allowed_text = (job_text + "\n" + resume_context).lower()
        
        # Repair evidence
        chunk_tokens = [_tokenize(d.page_content) for d in chunks]
        # (Using imported _best_chunk_ids requires passing it the helper or closures, 
        # actually _best_chunk_ids in resume_tuner relies on closure `chunk_tokens`. 
        # I should probably just reimplement the small logic here to be safe and self-contained 
        # or update resume_tuner to be more functional. 
        # For speed, I will implement the logic inline here.)
        
        def local_best_chunk_ids(text, top_n=2):
            q = _tokenize(text)
            if not q: return []
            scored = []
            for idx, toks in enumerate(chunk_tokens, start=1):
                overlap = len(q & toks)
                if overlap > 0:
                    scored.append((overlap, idx))
            scored.sort(reverse=True)
            return [f"CHUNK {idx}" for _, idx in scored[:top_n]] if scored else []

        # Fix evidence chunks
        if isinstance(model_output, dict):
            for key, text_field in (("gap_analysis", "gap"), ("rewrite_suggestions", "suggestion"), ("latex_bullets", "bullet")):
                items = model_output.get(key)
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            text_val = item.get(text_field)
                            if isinstance(text_val, str) and text_val.strip():
                                ev = item.get("evidence_chunks")
                                if not isinstance(ev, list) or len(ev) == 0:
                                    item["evidence_chunks"] = local_best_chunk_ids(text_val)
                                if key == "latex_bullets":
                                    bullet = item.get("bullet", "")
                                    if isinstance(bullet, str) and bullet.strip() and not bullet.strip().startswith("\\item"):
                                        item["bullet"] = "\\item " + bullet.strip()

        # Guardrails
        guardrail_removed = []
        if isinstance(model_output, dict):
             # tech_token_re is not imported, define it
            tech_token_re = re.compile(
                r"\b("
                r"[A-Z]{2,}"
                r"|[A-Z][a-z]+[A-Z][A-Za-z0-9]*"
                r"|[A-Z][A-Za-z0-9]*(?:[+._-][A-Za-z0-9]+)+"
                r"|[A-Za-z]+\d+[A-Za-z0-9.]*"
                r")\b"
            )
            
            for key in ("gap_analysis", "rewrite_suggestions", "latex_bullets"):
                items = model_output.get(key)
                if not isinstance(items, list): continue
                kept = []
                for item in items:
                    if not isinstance(item, dict): continue
                    
                    if key == "gap_analysis": text = item.get("gap")
                    elif key == "rewrite_suggestions": text = item.get("suggestion")
                    else: text = item.get("bullet")
                    
                    if isinstance(text, str):
                        # Unseen tech check
                        tokens = sorted(set(tech_token_re.findall(text or "")))
                        unseen = [t for t in tokens if t.lower() not in allowed_text]
                        if unseen:
                            guardrail_removed.append({"type": key, "item": item, "unseen_tech": unseen})
                            continue
                        
                        # Novelty check (using imported _novelty_ratio if simple enough, but it needs allowed_tokens context)
                        # Let's just implement simply
                        toks = _tokenize(text)
                        allowed_toks_set = _tokenize(allowed_text)
                        if toks:
                            novel = toks - allowed_toks_set
                            ratio = len(novel) / max(len(toks), 1)
                            if ratio > 0.80:
                                guardrail_removed.append({"type": key, "item": item, "reason": "high_novelty"})
                                continue
                        
                        # Contradiction check
                        if key == "gap_analysis" and re.search(r"\b(lack|lacks|missing)\b", text, flags=re.IGNORECASE):
                             mentioned = sorted(set(tech_token_re.findall(text)))
                             if any(m.lower() in allowed_text for m in mentioned):
                                 guardrail_removed.append({"type": key, "item": item, "reason": "contradicts_inputs"})
                                 continue
                    
                    kept.append(item)
                model_output[key] = kept
            
            if guardrail_removed:
                model_output["_guardrails"] = {"removed_count": len(guardrail_removed), "removed": guardrail_removed}
            
            # Derive bullets if missing
            if isinstance(model_output.get("latex_bullets"), list) and not model_output["latex_bullets"]:
                rs = model_output.get("rewrite_suggestions")
                if isinstance(rs, list) and rs:
                    derived = []
                    for item in rs:
                        if isinstance(item, dict):
                            s = item.get("suggestion")
                            if isinstance(s, str) and s.strip():
                                derived.append({"bullet": "\\item " + s.strip(), "evidence_chunks": item.get("evidence_chunks", [])})
                    model_output["latex_bullets"] = derived


        return JSONResponse(content={
            "status": "success",
            "retrieval_stats": retrieval_stats,
            "analysis": model_output
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Mount frontend AFTER API routes to avoid capturing them
    uvicorn.run(app, host="0.0.0.0", port=8000)

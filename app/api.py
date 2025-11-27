import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import run_agent
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Career Coach API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeRequest(BaseModel):
    job_url: str

@app.post("/generate-resume")
async def generate_resume(request: ResumeRequest):
    """
    Triggers the AI Agent to analyze the job, tune the resume, and find similar jobs.
    """
    try:
        print(f"Received request for URL: {request.job_url}")
        # run_agent returns the structured output from the LLM
        result = run_agent(request.job_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("output", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/pdf", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


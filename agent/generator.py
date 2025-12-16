import json
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
# import pdfkit  # Only works if wkhtmltopdf is installed system-wide

# Ideally we'd use a robust PDF library or a headless browser (playwright)
# For simplicity, we'll generate HTML and let the user Print to PDF, 
# or use a python PDF library like WeasyPrint if installed.
# For this step, let's just output the HTML file.

TEMPLATES_DIR = Path("templates")
OUTPUT_DIR = Path("generated_resumes")
OUTPUT_DIR.mkdir(exist_ok=True)

class ResumeGenerator:
    def __init__(self):
        self.env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        self.template = self.env.get_template("resume_template.html")

    def generate(self, data: dict, filename: str = "tuned_resume.html") -> str:
        """
        Render the HTML resume from data dict.
        """
        html_content = self.template.render(**data)
        
        output_path = OUTPUT_DIR / filename
        with open(output_path, "w") as f:
            f.write(html_content)
        
        return str(output_path)

    def data_from_chunks(self, chunks, gap_analysis, verified_skills):
        """
        Heuristic to rebuild a structured resume object from retrieved chunks 
        PLUS the new gap analysis content.
        
        Realistically, to rebuild the *entire* resume perfectly, we need to parse 
        the original PDF into structured JSON first. RAG chunks are fragmented.
        
        STRATEGY:
        1. Parse the *Original* PDF into a structured dict (using an LLM call or struct parser).
        2. Inject the *Verified Skills* into the Skills section.
        3. Inject the *Rewrite Suggestions* into the Experience section.
        """
        # Placeholder for structured data
        # In a real app, we'd have a 'ResumeParser' that converts PDF -> JSON at ingest time.
        
        base_data = {
            "name": "Candidate Name", # Placeholder
            "email": "candidate@email.com",
            "phone": "123-456-7890",
            "location": "City, Country",
            "experience": [
                {
                    "company": "Company A",
                    "title": "Software Engineer",
                    "dates": "2020 - Present",
                    "bullets": [
                        "Original bullet point 1",
                        "Original bullet point 2"
                    ]
                }
            ],
            "skills": {
                "languages": "Python, JavaScript",
                "frameworks": "React, FastAPI",
                "tools": "Git, Docker"
            },
            "education": [
                {
                    "institution": "University X",
                    "degree": "B.Sc. Computer Science",
                    "dates": "2016 - 2020"
                }
            ]
        }
        
        # Inject Verified Skills
        if verified_skills:
            # Naively append to 'tools' or create a new category
            base_data["skills"]["tools"] += ", " + ", ".join(verified_skills)
            
        return base_data

generator = ResumeGenerator()


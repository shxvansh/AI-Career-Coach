import os
import re
from pathlib import Path
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

# Import our RAG/LoRA modules
# Note: We assume these modules are available in python path
from RAG.query import retrieve_resume_chunks_from_text
from resume_tuner import _build_resume_context
from finetuning.inference import ResumeExpert

# Load environment variables
load_dotenv()

PERSIST_DIR = Path("tuner_db")
ADAPTER_PATH = Path("finetuning/resume-expert-lora")

class CareerAgent:
    def __init__(self):
        self.memory: List[Any] = []
        self.state = "INIT"  # INIT, RESEARCH, GAP_ANALYSIS, VERIFICATION, GENERATION
        self.context = {
            "resume_text": "",
            "job_text": "",
            "job_source": "", # Path to job PDF
            "web_research": "",
            "missing_skills": [],
            "verified_skills": [],
            "gap_analysis_result": None
        }
        
        # Tools
        self.tavily = TavilySearchResults(max_results=3)
        self.expert_model = None

    def _get_model(self):
        if self.expert_model is None:
            if ADAPTER_PATH.exists():
                print(f"Loading Resume Expert from {ADAPTER_PATH}")
                self.expert_model = ResumeExpert(adapter_path=str(ADAPTER_PATH))
            else:
                print("Warning: Adapter not found, loading base model.")
                self.expert_model = ResumeExpert(adapter_path=None)
        return self.expert_model

    def add_message(self, role: str, content: str):
        if role == "user":
            self.memory.append(HumanMessage(content=content))
        else:
            self.memory.append(AIMessage(content=content))

    def _generate_resume(self) -> str:
        # Import Generator
        from agent.generator import generator
        
        # 1. Parse Resume (Naive placeholder or we can use PyMuPDF to get text struct)
        # For this prototype, we rely on the heuristic generator.data_from_chunks
        # but to make it real, we need structured data.
        # Let's rely on the dummy data in generator.py for now, injected with real verified skills.
        
        # In a real implementation:
        # parsed_resume = self._parse_resume(self.context["resume_text"])
        
        final_data = generator.data_from_chunks(
            chunks=[], # We don't have structured chunks yet
            gap_analysis=self.context.get("gap_analysis_result"),
            verified_skills=self.context.get("verified_skills", [])
        )
        
        # Add a name from filename if possible
        if self.context.get("resume_text"):
             # Heuristic: First line is name
             final_data["name"] = self.context["resume_text"].split('\n')[0].strip()
        
        output_path = generator.generate(final_data)
        
        return f"Resume generated successfully! You can download it at: {output_path}"

    def run(self, user_input: str) -> str:
        self.add_message("user", user_input)
        
        # Simple State Machine
        if self.state == "INIT":
            if self.context.get("job_text"):
                 self.state = "RESEARCH"
                 return self._perform_research()
            response = "Hello! I am your AI Career Coach. Please upload your Resume and the Job Description PDF to get started."
            self.add_message("ai", response)
            return response

        elif self.state == "RESEARCH":
            if not self.context.get("job_text"):
                self.state = "INIT"
                return "Please upload the files first."
            return self._perform_research()

        elif self.state == "GAP_ANALYSIS":
            return self._perform_gap_analysis()

        elif self.state == "VERIFICATION":
            return self._handle_verification(user_input)

        elif self.state == "GENERATION":
            return self._generate_resume()

        return "I'm not sure what to do next."

    def _extract_keywords(self, text: str) -> str:
        # Simple heuristic to get job title/company
        # "Machine Learning Engineer at Google"
        # We can try to extract the first few lines or look for "Company"
        lines = text.split('\n')[:5]
        return " ".join(lines).strip()[:100]

    def _perform_research(self) -> str:
        job_text = self.context.get("job_text", "")
        if not job_text:
            return "Error: Job text missing."
            
        # Extract query
        # Use first 200 chars as query proxy or extract title
        query = f"Job requirements for: {self._extract_keywords(job_text)}"
        
        try:
            results = self.tavily.invoke(query)
            summary = "\n".join([f"- {r['content']}" for r in results])
            self.context["web_research"] = summary
            
            response = (
                f"I've researched the role and company online. Here are some key insights:\n\n"
                f"{summary}\n\n"
                "I will now cross-reference this with your resume to find gaps."
            )
            self.state = "GAP_ANALYSIS"
            self.add_message("ai", response)
            
            # Auto-trigger gap analysis? 
            # In a chat loop, usually we wait for user, but here the agent is driving.
            # Let's return the research and immediately hint we are moving to gap analysis.
            # Or better, just do it in next turn? 
            # Let's auto-advance state but return this message. 
            # The API endpoint loop checks state. We can chain calls if we want.
            # For now, return this, and user says "Okay" or "Proceed".
            return response
        except Exception as e:
            return f"Error during research: {e}"

    def _perform_gap_analysis(self) -> str:
        job_text = self.context.get("job_text", "")
        web_research = self.context.get("web_research", "")
        
        # Augmented Job Description
        augmented_job_text = f"{job_text}\n\n=== WEB RESEARCH INSIGHTS ===\n{web_research}"
        
        # RAG Retrieval
        chunks, _ = retrieve_resume_chunks_from_text(
            job_text=augmented_job_text,
            persist_dir=PERSIST_DIR,
            k=4,
            job_source=self.context.get("job_source", "uploaded_job.pdf")
        )
        
        if not chunks:
            return "I couldn't find any relevant sections in your resume. Did you upload it?"

        resume_context = _build_resume_context(chunks)
        
        # LoRA Inference
        model = self._get_model()
        analysis = model.generate_grounded_json(
            resume_context=resume_context,
            job_description=augmented_job_text
        )
        
        self.context["gap_analysis_result"] = analysis
        
        # Extract missing keywords from analysis
        # The model returns "missing_keywords" (list of dicts) or "gap_analysis" (list of dicts)
        missing = []
        if isinstance(analysis, dict):
            # Check missing_keywords
            mk = analysis.get("missing_keywords", [])
            for m in mk:
                if isinstance(m, dict) and m.get("keyword"):
                    missing.append(m.get("keyword"))
            
            # Check gap_analysis gaps
            ga = analysis.get("gap_analysis", [])
            for g in ga:
                if isinstance(g, dict) and g.get("gap"):
                     # Heuristic: extract the skill from the gap text if possible
                     # For now just present the gap text
                     pass

        self.context["missing_skills"] = missing
        
        if not missing:
            # If no obvious missing keywords, just show gaps
            gaps_text = "\n".join([f"- {g.get('gap')}" for g in analysis.get('gap_analysis', [])])
            response = (
                "Good news! I didn't find any specific missing keywords. However, here are some gaps/weaknesses identified:\n"
                f"{gaps_text}\n\n"
                "Shall I proceed to generate the resume?"
            )
            self.state = "GENERATION" # Skip verification
        else:
            response = (
                "Based on the analysis, your resume is missing these key skills/topics:\n"
                f"- {', '.join(missing)}\n\n"
                "**Verification Required**: To prevent fraud, please tell me which of these you actually have experience with. "
                "For example: 'I have used Kubernetes for 2 years.' or 'Skip Kubernetes, I don't know it.'"
            )
            self.state = "VERIFICATION"

        self.add_message("ai", response)
        return response

    def _handle_verification(self, user_input: str) -> str:
        # Simple logic: Check which missing skills are mentioned in user_input
        # In a real agent, we'd use an LLM to classify "Yes/No" per skill.
        
        approved = []
        missing = self.context.get("missing_skills", [])
        
        lower_input = user_input.lower()
        
        # Very naive check
        if "skip" in lower_input and "all" in lower_input:
            pass # None approved
        elif "yes" in lower_input and "all" in lower_input:
            approved = missing
        else:
            for skill in missing:
                if skill.lower() in lower_input:
                    approved.append(skill)
        
        self.context["verified_skills"] = approved
        
        # Advance state
        self.state = "GENERATION"
        
        # Immediately generate
        gen_msg = self._generate_resume()
        
        response = (
            f"Understood. I will add the following skills: {', '.join(approved) if approved else 'None'}.\n\n"
            f"{gen_msg}"
        )
        self.add_message("ai", response)
        return response

agent_instance = CareerAgent()



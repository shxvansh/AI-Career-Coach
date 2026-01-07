import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
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
        self.context = {
            "has_resume": False,
            "job_text": None,
            "job_source": None,
            "last_gap_analysis": None
        }
        
        # Tools
        self.tavily = None  # Initialize on demand
        self.expert_model = None

    def _get_model(self):
        if self.expert_model is None:
            # Check if Gemini API key is available
            use_gemini = bool(os.getenv("GEMINI_API_KEY"))
            
            if use_gemini:
                print("Using Gemini API for Agent inference")
                self.expert_model = ResumeExpert(use_gemini=True)
            elif ADAPTER_PATH.exists():
                print(f"Loading Resume Expert from {ADAPTER_PATH}")
                self.expert_model = ResumeExpert(adapter_path=str(ADAPTER_PATH))
            else:
                print("Warning: Adapter not found, loading base model.")
                self.expert_model = ResumeExpert(adapter_path=None)
        return self.expert_model
    
    def _get_tavily(self):
        """Initialize Tavily search on demand"""
        if self.tavily is None:
            try:
                # Check if API key exists
                if not os.getenv("TAVILY_API_KEY"):
                    print("Info: TAVILY_API_KEY not set. Web search will be unavailable.")
                    return None
                self.tavily = TavilySearchResults(max_results=3)
            except Exception as e:
                print(f"Warning: Could not initialize Tavily search: {e}")
                self.tavily = None
        return self.tavily

    def add_message(self, role: str, content: str):
        if role == "user":
            self.memory.append(HumanMessage(content=content))
        else:
            self.memory.append(AIMessage(content=content))

    def _classify_intent(self, text: str) -> Dict[str, Any]:
        """
        Intelligently classifies user intent and extracts relevant information.
        Returns a dict with 'type' and additional context.
        """
        text_lower = text.lower()
        text_len = len(text)
        
        # Check for greetings
        greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']
        if any(text_lower.strip().startswith(g) for g in greetings) and text_len < 50:
            return {"type": "greeting"}
        
        # Check for job description (long text with JD keywords)
        jd_keywords = ['responsibilities', 'requirements', 'qualifications', 'job description', 
                       'salary', 'benefits', 'position', 'role', 'duties', 'apply', 'candidate',
                       'experience required', 'skills required', 'preferred qualifications']
        jd_score = sum(1 for k in jd_keywords if k in text_lower)
        
        if (text_len > 300 and jd_score >= 2) or jd_score >= 4:
            return {"type": "job_description", "text": text}
        
        # Check for resume questions
        resume_keywords = ['my resume', 'my cv', 'my experience', 'my skills', 'my background',
                          'resume', 'cv', 'experiences', 'qualifications']
        if any(k in text_lower for k in resume_keywords):
            return {"type": "resume_question", "query": text}
        
        # Check for analysis requests
        analysis_keywords = ['analyze', 'compare', 'gap', 'strength', 'weakness', 'match',
                            'fit', 'suitable', 'qualified']
        if any(k in text_lower for k in analysis_keywords):
            if self.context.get("job_text"):
                return {"type": "analysis_request"}
            else:
                return {"type": "analysis_request_no_jd"}
        
        # Check if it's a general question (needs web search)
        question_indicators = ['what is', 'what are', 'how to', 'why', 'when', 'where',
                              'explain', 'tell me about', 'describe']
        if any(q in text_lower for q in question_indicators) or '?' in text:
            # If has resume context, it's likely about resume
            if self.context["has_resume"]:
                return {"type": "resume_question", "query": text}
            else:
                return {"type": "general_question", "query": text}
        
        # Default: general chat
        return {"type": "general_chat", "message": text}
    
    def run_stream(self, user_input: str):
        """
        Main conversational handler - routes to appropriate functions based on intent.
        Yields chunks of text.
        """
        self.add_message("user", user_input)
        
        # Classify intent
        intent = self._classify_intent(user_input)
        intent_type = intent.get("type")
        
        print(f"DEBUG: Intent detected - {intent_type}")
        
        # Route to appropriate handler
        generator = None
        if intent_type == "greeting":
            generator = self._handle_greeting_stream()
        
        elif intent_type == "job_description":
            generator = self._handle_job_description_stream(intent["text"])
        
        elif intent_type == "resume_question":
            generator = self._handle_resume_question_stream(intent["query"])
        
        elif intent_type == "analysis_request":
            generator = self._handle_analysis_request_stream()
        
        elif intent_type == "analysis_request_no_jd":
            def _gen(): yield "I'd be happy to analyze your resume! Please provide the job description you'd like to compare it against."
            generator = _gen()
        
        elif intent_type == "general_question":
            generator = self._handle_general_question_stream(intent["query"])
        
        elif intent_type == "general_chat":
            generator = self._handle_general_chat_stream(intent["message"])
        
        else:
            generator = self._handle_general_chat_stream(user_input)
            
        # Consume generator, yield chunks, and accumulate for history
        full_response = ""
        for chunk in generator:
            full_response += chunk
            yield chunk
            
        # Add final full response to memory
        self.add_message("ai", full_response)

    def run(self, user_input: str) -> str:
        """Legacy synchronous run"""
        response = ""
        for chunk in self.run_stream(user_input):
            response += chunk
        return response

    def _handle_greeting_stream(self):
        """Handle greetings"""
        if self.context["has_resume"]:
            yield "Hello! I have your resume loaded. How can I help you today? You can ask me about your resume, provide a job description for analysis, or ask me general career questions."
        else:
            yield "Hello! I'm your AI Career Coach. Please upload your resume to get started, and I'll be happy to help you with career advice, resume analysis, and job matching!"
    
    def _handle_job_description_stream(self, job_text: str):
        """Handle when user provides a job description"""
        if not self.context["has_resume"]:
            yield "I'd be happy to analyze this job description against your resume, but I don't have your resume yet. Please upload it first!"
            return
        
        # Store the job description
        self.context["job_text"] = job_text
        
        # Perform analysis
        try:
            # Analysis is currently monolithic (not streaming), so we yield the result as one chunk
            analysis_result = self._perform_gap_analysis(job_text)
            yield analysis_result
        except Exception as e:
            yield f"I encountered an error analyzing the job description: {str(e)}"
    
    def _handle_resume_question_stream(self, query: str):
        """Handle questions about the user's resume"""
        if not self.context["has_resume"]:
            yield "I don't have your resume loaded yet. Please upload it first so I can answer questions about it!"
            return
        
        try:
            # Retrieve relevant chunks from resume
            chunks, _ = retrieve_resume_chunks_from_text(
                job_text=query,
                persist_dir=PERSIST_DIR,
                k=3,
                job_source="chat_query"
            )
            
            if not chunks:
                yield "I couldn't find relevant information in your resume to answer that question. Could you rephrase or ask something else?"
                return
            
            context_str = "\n\n".join([c.page_content for c in chunks])
            
            # Use LLM to answer
            model = self._get_model()
            prompt = f"""You are an AI Career Coach helping someone with their resume.

Resume Context (relevant sections):
{context_str}

User Question: {query}

Instructions:
1. Answer the question using ONLY information from the Resume Context above
2. Be conversational, friendly, and helpful
3. If the information isn't in the resume context, say so politely
4. Keep your response concise but complete
5. Respond in Markdown.
6. If you provide strengths/weaknesses, use exactly these headings: "### Strengths" and "### Weaknesses", then use bullet points starting with "- ".
7. Use **bold** for key phrases, and keep lines short for readability.

Answer:"""
            
            # Stream response
            for chunk in model.chat_stream(prompt):
                yield chunk
            
        except Exception as e:
            yield f"I had trouble accessing your resume information: {str(e)}"
    
    def _handle_analysis_request_stream(self):
        """Handle explicit requests for gap/strength/weakness analysis"""
        if not self.context["has_resume"]:
            yield "I need your resume to perform an analysis. Please upload it first!"
            return
        
        if not self.context.get("job_text"):
            yield "I'd be happy to analyze your resume! Could you provide the job description you'd like me to compare it against?"
            return
        
        # Perform the analysis (monolithic)
        try:
            yield self._perform_gap_analysis(self.context["job_text"])
        except Exception as e:
            yield f"I encountered an error during analysis: {str(e)}"
    
    def _handle_general_question_stream(self, query: str):
        """Handle general career questions using web search + LLM"""
        try:
            # Use web search to get current information
            tavily = self._get_tavily()
            if tavily:
                search_results = tavily.invoke(query)
                context = "\n\n".join([f"Source: {r.get('url', 'Unknown')}\n{r.get('content', '')}" 
                                      for r in search_results])
                
                model = self._get_model()
                prompt = f"""You are an AI Career Coach. Answer the user's question using the web search results provided.

Web Search Results:
{context}

User Question: {query}

Instructions:
1. Provide a helpful, accurate answer based on the search results
2. Be conversational and professional
3. If relevant, relate it to career advice
4. Keep it concise but informative
5. Respond in Markdown. Prefer bullet points and short paragraphs. Use **bold** for key phrases.

Answer:"""
                
                for chunk in model.chat_stream(prompt):
                    yield chunk
            else:
                # Fallback if Tavily is not available
                model = self._get_model()
                for chunk in model.chat_stream(f"As a career coach, answer this question: {query}"):
                    yield chunk
                
        except Exception as e:
            yield f"I had trouble researching that question: {str(e)}"
    
    def _handle_general_chat_stream(self, message: str):
        """Handle general conversation"""
        model = self._get_model()
        
        # Build context from recent conversation
        recent_history = ""
        if len(self.memory) > 1:
            recent_messages = self.memory[-4:]  # Last 4 messages
            for msg in recent_messages:
                if isinstance(msg, HumanMessage):
                    recent_history += f"User: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    recent_history += f"Assistant: {msg.content}\n"
        
        prompt = f"""You are an AI Career Coach chatting with a user. Be friendly, professional, and helpful.

{f'Recent conversation:{recent_history}' if recent_history else ''}

User: {message}

Respond naturally and helpfully. If appropriate, guide them toward career-related assistance you can provide (resume review, job matching, career advice).
Respond in Markdown. Prefer bullet points when listing multiple items. Use **bold** for emphasis.

Response:"""
        
        try:
            for chunk in model.chat_stream(prompt):
                yield chunk
        except Exception as e:
            yield "I'm here to help with your career! You can ask me about your resume, provide job descriptions for analysis, or ask career-related questions."

    
    def _perform_gap_analysis(self, job_text: str) -> str:
        """Perform comprehensive gap analysis with optional web research"""
        if not self.context["has_resume"]:
            return "I need your resume to perform this analysis."
        
        # Optional: Do web research for more context about the role
        web_research = ""
        try:
            tavily = self._get_tavily()
            if tavily:
                # Extract key terms for search
                lines = job_text.split('\n')[:5]
                search_query = " ".join(lines)[:150]
                
                results = tavily.invoke(f"Job requirements skills: {search_query}")
                web_research = "\n".join([f"- {r.get('content', '')}" for r in results])
        except Exception as e:
            print(f"Web research failed: {e}")
        
        # Augment job description with web research if available
        augmented_job = job_text
        if web_research:
            augmented_job = f"{job_text}\n\n=== Additional Industry Context ===\n{web_research}"
        
        # Retrieve relevant resume chunks
        try:
            chunks, _ = retrieve_resume_chunks_from_text(
                job_text=augmented_job,
                persist_dir=PERSIST_DIR,
                k=4,
                job_source="chat_analysis"
            )
            
            if not chunks:
                return "I couldn't find relevant information in your resume. Make sure it was uploaded correctly."
            
            resume_context = _build_resume_context(chunks)
            
            # Generate analysis using the model
            model = self._get_model()
            analysis = model.generate_grounded_json(
                resume_context=resume_context,
                job_description=augmented_job
            )
            
            # Store for later reference
            self.context["last_gap_analysis"] = analysis
            
            # Format response
            response = self._format_analysis_response(analysis)
            # self.add_message("ai", response) # Handled by run_stream
            return response
            
        except Exception as e:
            return f"Error during analysis: {str(e)}"
    
    def _format_analysis_response(self, analysis: Dict[str, Any]) -> str:
        """Format the analysis result into a readable response"""
        if not isinstance(analysis, dict):
            return "I completed the analysis but had trouble formatting the results."
        
        response_parts = []
        
        # Strengths (from matching keywords or positive points)
        strengths = []
        if "matching_keywords" in analysis:
            for item in analysis["matching_keywords"][:5]:
                if isinstance(item, dict):
                    strengths.append(item.get("keyword", ""))
        
        if strengths:
            response_parts.append("### Strengths")
            response_parts.append("**Skills that match well:**")
            for s in strengths:
                if s:
                    response_parts.append(f"- {s}")
            response_parts.append("")  # spacer
        
        # Gaps/Weaknesses
        gaps = []
        if "gap_analysis" in analysis:
            for item in analysis["gap_analysis"][:5]:
                if isinstance(item, dict):
                    gaps.append(item.get("gap", ""))
        
        if gaps:
            response_parts.append("### Weaknesses / Gaps")
            response_parts.append("**Areas for improvement (based on the JD):**")
            for g in gaps:
                if g:
                    response_parts.append(f"- {g}")
            response_parts.append("")  # spacer
        
        # Missing keywords
        missing = []
        if "missing_keywords" in analysis:
            for item in analysis["missing_keywords"][:5]:
                if isinstance(item, dict):
                    missing.append(item.get("keyword", ""))
        
        if missing:
            response_parts.append("### Missing keywords (only add if true)")
            for m in missing:
                if m:
                    response_parts.append(f"- {m}")
            response_parts.append("")  # spacer
        
        if not response_parts:
            return "I've analyzed your resume against the job description. Overall, you seem to have a good match!"
        
        return "\n".join(response_parts)

agent_instance = CareerAgent()



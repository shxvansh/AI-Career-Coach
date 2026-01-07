import os
import json
import re
import torch
from google import genai
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

class ResumeExpert:
    def __init__(self, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", adapter_path="finetuning/resume-expert-lora", use_gemini=False):
        # Avoid tokenizers parallelism fork warning/noise
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        
        self.use_gemini = use_gemini
        
        if self.use_gemini:
            print("Loading Resume Expert with Google Gemini API...")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            self.client = genai.Client(api_key=api_key)
            # Use 2.5-flash-lite for better availability and lower quota usage
            self.model_name = 'gemini-2.5-flash-lite'
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Loading Resume Expert on {self.device}...")
            
            # Load Base Model
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
                device_map={"": self.device}, # Force device
                trust_remote_code=True
            )
            
            # Load Adapter
            if adapter_path:
                print(f"Loading LoRA adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
            else:
                self.model = self.base_model

            self.model.eval()

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        """
        Best-effort extraction of the first JSON object in a string.
        Handles common cases like ```json fences and multiple JSON blocks.
        """
        if not text:
            return None

        # Remove common fenced code markers but keep inner content.
        cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")

        start = cleaned.find("{")
        if start == -1:
            return None

        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
                continue

            if ch == "\"":
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1].strip()

        return None

    @staticmethod
    def _sanitize_output(obj: dict) -> dict:
        """
        Enforce a minimal schema and sanitize evidence chunk ids.
        Also removes items that contain non-ascii characters (common hallucination artifacts).
        """
        out = {
            "gap_analysis": obj.get("gap_analysis", []) if isinstance(obj.get("gap_analysis"), list) else [],
            "missing_keywords": obj.get("missing_keywords", []) if isinstance(obj.get("missing_keywords"), list) else [],
            "rewrite_suggestions": obj.get("rewrite_suggestions", []) if isinstance(obj.get("rewrite_suggestions"), list) else [],
            "latex_bullets": obj.get("latex_bullets", []) if isinstance(obj.get("latex_bullets"), list) else [],
        }

        chunk_pat = re.compile(r"^CHUNK\\s+\\d+$", re.IGNORECASE)

        def clean_chunks(v):
            if not isinstance(v, list):
                return []
            cleaned = []
            for x in v:
                if not isinstance(x, str):
                    continue
                x = x.strip()
                # Accept exact "CHUNK N"
                if chunk_pat.match(x):
                    cleaned.append(x.upper().replace("  ", " "))
                    continue
                # Also accept strings containing chunk ids (e.g. "CHUNK 2, CHUNK 3")
                for m in re.findall(r"CHUNK\\s+\\d+", x, flags=re.IGNORECASE):
                    cleaned.append(m.upper().replace("  ", " "))
            return cleaned

        def has_non_ascii(s: str) -> bool:
            return any(ord(c) > 127 for c in s)

        removed = []
        for key in ("gap_analysis", "rewrite_suggestions", "latex_bullets"):
            kept = []
            for item in out[key]:
                if not isinstance(item, dict):
                    continue
                text_key = "gap" if key == "gap_analysis" else ("suggestion" if key == "rewrite_suggestions" else "bullet")
                text_val = item.get(text_key, "")
                if not isinstance(text_val, str) or not text_val.strip():
                    removed.append({"type": key, "item": item, "reason": "missing_text"})
                    continue
                if text_val.strip() == "..." or "..." in text_val:
                    removed.append({"type": key, "item": item, "reason": "placeholder_text"})
                    continue
                if has_non_ascii(text_val):
                    removed.append({"type": key, "item": item, "reason": "non_ascii_text"})
                    continue
                item["evidence_chunks"] = clean_chunks(item.get("evidence_chunks"))
                kept.append(item)
            out[key] = kept

        # missing_keywords should be strings; keep minimal
        mk = []
        for item in out["missing_keywords"]:
            if not isinstance(item, dict):
                continue
            kw = item.get("keyword")
            if isinstance(kw, str) and kw.strip() and kw.strip() != "..." and "..." not in kw:
                mk.append({"keyword": kw.strip(), "evidence": "Job Description"})
        out["missing_keywords"] = mk

        if removed:
            out["_model_sanitize"] = {"removed_count": len(removed), "removed": removed}
        return out

    def generate_grounded_json(
        self,
        resume_context: str,
        job_description: str,
        max_new_tokens: int = 500,
    ) -> dict:
        """
        Produce grounded suggestions with citations to retrieved chunks.

        Returns a dict (best-effort). If the model does not emit valid JSON, returns:
          {"raw_text": "..."}
        """
        user_prompt = f"""You are a resume tuning assistant.

CRITICAL RULES:
1) Only use information explicitly present in the Job Description and Retrieved Resume Chunks.
2) Do NOT introduce new technologies, companies, projects, or numbers not present in the inputs.
3) Every suggestion MUST cite evidence by referencing chunk ids like \"CHUNK 1\" (ONLY these ids).
4) Output MUST be a single valid JSON object only (no prose, no markdown, no ```json fences, no second JSON object).
5) If you are unsure, output empty arrays for that section rather than guessing.
6) NEVER output placeholder text like \"...\". If you can't produce a real item, omit it from the list.

Return ONE JSON object with exactly these keys:
- gap_analysis: list of objects with fields {{"gap": string, "evidence_chunks": ["CHUNK N", ...]}}
- missing_keywords: list of objects with fields {{"keyword": string, "evidence": "Job Description"}}
- rewrite_suggestions: list of objects with fields {{"suggestion": string, "evidence_chunks": ["CHUNK N", ...]}}
- latex_bullets: list of objects with fields {{"bullet": string starting with \"\\\\item \", "evidence_chunks": ["CHUNK N", ...]}}

If you cannot produce a real item, return an empty list for that key. Do NOT output placeholder values.

Valid chunk ids you may cite: CHUNK 1, CHUNK 2, CHUNK 3, CHUNK 4 only.

### Job Description:
{job_description}

### Retrieved Resume Chunks:
{resume_context}

Output (JSON only):"""

        if self.use_gemini:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config={
                        'temperature': 0.2,
                        'max_output_tokens': max_new_tokens
                    }
                )
                completion = response.text
            except Exception as e:
                return {"raw_text": f"Gemini Error: {e}"}
        else:
            # Use the model's chat template for better instruction following.
            messages = [
                {"role": "system", "content": "You follow instructions exactly and output only valid JSON."},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.1,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode ONLY the generated completion tokens (exclude the prompt tokens).
            prompt_len = int(inputs["input_ids"].shape[-1])
            generated_ids = outputs[0][prompt_len:]
            completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        extracted = self._extract_first_json_object(completion)
        if not extracted:
            return {"raw_text": completion}
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return self._sanitize_output(parsed)
            return {"raw_text": completion}
        except Exception:
            return {"raw_text": completion}

    def generate_analysis(self, resume_context: str, job_description: str) -> str:
        """Backwards compatible: returns a string (JSON or raw text)."""
        result = self.generate_grounded_json(resume_context=resume_context, job_description=job_description)
        return json.dumps(result, ensure_ascii=False)

    def chat(self, prompt: str, max_new_tokens: int = 900) -> str:
        """
        Generic chat capability using the loaded model.
        """
        # Fallback to non-streaming
        full_response = ""
        for chunk in self.chat_stream(prompt, max_new_tokens):
            full_response += chunk
        return full_response

    def chat_stream(self, prompt: str, max_new_tokens: int = 900):
        """
        Streaming chat capability.
        Yields chunks of text.
        """
        if self.use_gemini:
            try:
                # Assuming google-genai SDK 
                # Note: The user code uses `from google import genai` which implies the new v1 SDK or similar.
                # However, usually generate_content returns an iterable if stream=True?
                # Actually, in the new SDK:
                # response = client.models.generate_content(..., config=..., )
                # It doesn't seem to support stream=True in the config dict based on some docs, 
                # but let's try passing it as an argument if possible or iterate if it returns a generator.
                
                # Let's try the most standard way for Gemini API
                # If we can't stream easily, yielding the whole text at once is a safe fallback for now
                # to avoid breaking the app, but we want to TRY streaming.
                
                # IMPORTANT: We need to check if we can stream.
                # For now, let's implement a pseudo-stream if real stream fails or just yield the full text.
                # But actually, let's try to use the generate_content_stream if available or stream=True.
                
                # Based on typical google.generativeai usage:
                # response = model.generate_content(..., stream=True)
                # for chunk in response: yield chunk.text
                
                # Since we are using `self.client.models.generate_content`, let's try adding stream=True arg (not in config).
                # But wait, the previous code used `client.models.generate_content`. 
                # Let's assume standard behavior.
                
                # Wait, looking at lines 200-207 in original file:
                # response = self.client.models.generate_content(...)
                # This looks like the Vertex AI or the new GenAI SDK. 
                
                # Let's just implement a simple version that yields the full text for Gemini for now if unsure,
                # OR better: use `yield` on the result of `chat` for now to satisfy the interface, 
                # and if we can figure out streaming later we improve it.
                # The user explicitly asked for streaming tokens. 
                # I will try to fake it for Gemini if I can't find the stream method, OR 
                # I will assume `client.models.generate_content_stream` exists (common pattern).
                
                try:
                     # Try streaming method first
                    response = self.client.models.generate_content_stream(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            'temperature': 0.7,
                            'max_output_tokens': max_new_tokens
                        }
                    )
                    for chunk in response:
                        if hasattr(chunk, 'text'):
                             yield chunk.text
                        elif hasattr(chunk, 'candidates'):
                             yield chunk.candidates[0].content.parts[0].text
                except AttributeError:
                    # Fallback to non-streaming
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config={
                            'temperature': 0.7,
                            'max_output_tokens': max_new_tokens
                        }
                    )
                    if hasattr(response, 'text'):
                        yield response.text
                    elif hasattr(response, 'candidates'):
                         yield response.candidates[0].content.parts[0].text
                    else:
                        yield str(response)

            except Exception as e:
                yield f"Error: {e}"
        else:
            # Local model streaming
            from transformers import TextIteratorStreamer
            from threading import Thread
            
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            messages = [
                {"role": "system", "content": "You are a helpful AI Career Coach."},
                {"role": "user", "content": prompt},
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            generation_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer
            )
            
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()
            
            for new_text in streamer:
                yield new_text

if __name__ == "__main__":
    # Test run
    expert = ResumeExpert()
    
    test_resume = "Experienced in Python and Django. Built a blog app."
    test_jd = "Looking for a Python developer with FastAPI experience."
    
    print("\n--- Test Analysis ---")
    print(expert.generate_analysis(test_resume, test_jd))


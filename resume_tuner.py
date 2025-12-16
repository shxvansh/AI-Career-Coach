"""Resume tuner orchestrator.

Pipeline:
- Ingest resume PDF into Chroma (fresh persist dir)
- Extract job text
- Retrieve top-k resume chunks (default k=4)
- Run LoRA ResumeExpert to generate gap analysis + LaTeX bullets

Run:
  ./venv/bin/python resume_tuner.py --resume /path/to/resume.pdf --job /path/to/job.pdf
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from RAG.ingest import ingest_resume
from RAG.query import load_job_pdf, retrieve_resume_chunks_from_text
from finetuning.inference import ResumeExpert


def _build_resume_context(chunks, max_chars: int = 9000) -> str:
    parts: List[str] = []
    total = 0
    for i, doc in enumerate(chunks, start=1):
        section = str(doc.metadata.get("section_type", "other"))
        relevance = float(doc.metadata.get("relevance_score", 0.0) or 0.0)
        header = f"[CHUNK {i} | section={section} | relevance={relevance:.3f}]\n"
        body = (doc.page_content or "").strip()
        chunk_text = header + body
        if total + len(chunk_text) > max_chars:
            remaining = max_chars - total
            if remaining <= 0:
                break
            parts.append(chunk_text[:remaining])
            total += remaining
            break
        parts.append(chunk_text)
        total += len(chunk_text)
    return "\n\n".join(parts).strip()


def tune_resume(
    resume_pdf: Path,
    job_pdf: Path,
    persist_dir: Path,
    k: int = 4,
    adapter_path: Path = Path("finetuning/resume-expert-lora"),
) -> Dict[str, Any]:
    start = time.time()

    ingest_stats = ingest_resume(resume_path=resume_pdf, persist_dir=persist_dir)

    job_docs = load_job_pdf(job_pdf)
    job_text = "\n\n".join(d.page_content for d in job_docs).strip()

    chunks, retrieval_stats = retrieve_resume_chunks_from_text(
        job_text=job_text,
        persist_dir=persist_dir,
        k=k,
        job_source=str(job_pdf),
    )

    resume_context = _build_resume_context(chunks)

    expert = ResumeExpert(adapter_path=str(adapter_path))
    model_output = expert.generate_grounded_json(resume_context=resume_context, job_description=job_text)

    # Fill/repair evidence_chunks when the model doesn't comply.
    def _tokenize(s: str) -> set[str]:
        return {w for w in re.findall(r"[a-zA-Z0-9]+", (s or "").lower()) if len(w) >= 3}

    chunk_tokens = [_tokenize(d.page_content) for d in chunks]

    def _best_chunk_ids(text: str, top_n: int = 2) -> List[str]:
        q = _tokenize(text)
        if not q:
            return []
        scored = []
        for idx, toks in enumerate(chunk_tokens, start=1):
            overlap = len(q & toks)
            if overlap > 0:
                scored.append((overlap, idx))
        scored.sort(reverse=True)
        return [f"CHUNK {idx}" for _, idx in scored[:top_n]] if scored else []

    if isinstance(model_output, dict):
        for key, text_field in (
            ("gap_analysis", "gap"),
            ("rewrite_suggestions", "suggestion"),
            ("latex_bullets", "bullet"),
        ):
            items = model_output.get(key)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                text_val = item.get(text_field)
                if not isinstance(text_val, str) or not text_val.strip():
                    continue
                ev = item.get("evidence_chunks")
                if not isinstance(ev, list) or len(ev) == 0:
                    item["evidence_chunks"] = _best_chunk_ids(text_val, top_n=2)
                if key == "latex_bullets":
                    bullet = item.get("bullet", "")
                    if isinstance(bullet, str) and bullet.strip() and not bullet.strip().startswith("\\item"):
                        item["bullet"] = "\\item " + bullet.strip()

    # Minimal guardrail: drop suggestions that introduce tech terms not present in inputs.
    allowed_text = (job_text + "\n" + resume_context).lower()
    allowed_tokens = _tokenize(allowed_text)
    # Try to only match "tech-like" tokens (avoid sentence-leading verbs like "Integrated").
    # Examples matched: AWS, EC2, FastAPI, PyTorch, Qwen2.5, Node.js, XGBoost, Scikit-Learn
    tech_token_re = re.compile(
        r"\b("
        r"[A-Z]{2,}"  # acronyms (AWS, NLP)
        r"|[A-Z][a-z]+[A-Z][A-Za-z0-9]*"  # CamelCase (FastAPI, HuggingFace)
        r"|[A-Z][A-Za-z0-9]*(?:[+._-][A-Za-z0-9]+)+"  # punctuated w/ uppercase lead (Node.js, Scikit-Learn)
        r"|[A-Za-z]+\\d+[A-Za-z0-9.]*"  # contains digits (Qwen2.5)
        r")\b"
    )

    def _mentions_unseen_tech(text: str) -> List[str]:
        tokens = sorted(set(tech_token_re.findall(text or "")))
        unseen = [t for t in tokens if t.lower() not in allowed_text]
        return unseen

    def _novelty_ratio(text: str) -> float:
        toks = _tokenize(text)
        if not toks:
            return 0.0
        novel = toks - allowed_tokens
        return len(novel) / max(len(toks), 1)

    guardrail_removed: List[Dict[str, Any]] = []
    if isinstance(model_output, dict):
        for key in ("gap_analysis", "rewrite_suggestions", "latex_bullets"):
            items = model_output.get(key)
            if not isinstance(items, list):
                continue
            kept = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                if key == "gap_analysis":
                    text = item.get("gap")
                elif key == "rewrite_suggestions":
                    text = item.get("suggestion")
                else:
                    text = item.get("bullet")
                if isinstance(text, str):
                    unseen = _mentions_unseen_tech(text)
                    if unseen:
                        guardrail_removed.append({"type": key, "item": item, "unseen_tech": unseen})
                        continue
                    # If a suggestion is *almost entirely* novel text, drop it.
                    # (Keep this loose; the primary guardrail is "unseen tech".)
                    if _novelty_ratio(text) > 0.80:
                        guardrail_removed.append({"type": key, "item": item, "reason": "high_novelty"})
                        continue

                    # If a gap claims something is missing/lacking but that tech term is present in inputs, drop it.
                    if key == "gap_analysis" and re.search(r"\b(lack|lacks|missing)\b", text, flags=re.IGNORECASE):
                        mentioned = sorted(set(tech_token_re.findall(text)))
                        if any(m.lower() in allowed_text for m in mentioned):
                            guardrail_removed.append({"type": key, "item": item, "reason": "contradicts_inputs"})
                            continue
                kept.append(item)
            model_output[key] = kept
        if guardrail_removed:
            model_output["_guardrails"] = {
                "removed_count": len(guardrail_removed),
                "removed": guardrail_removed,
            }

        # If model didn't produce LaTeX bullets, derive them from rewrite suggestions.
        if isinstance(model_output.get("latex_bullets"), list) and not model_output["latex_bullets"]:
            rs = model_output.get("rewrite_suggestions")
            if isinstance(rs, list) and rs:
                derived = []
                for item in rs:
                    if not isinstance(item, dict):
                        continue
                    s = item.get("suggestion")
                    if not isinstance(s, str) or not s.strip():
                        continue
                    derived.append(
                        {
                            "bullet": "\\item " + s.strip(),
                            "evidence_chunks": item.get("evidence_chunks", []),
                        }
                    )
                model_output["latex_bullets"] = derived

    result = {
        "inputs": {
            "resume_pdf": str(resume_pdf),
            "job_pdf": str(job_pdf),
            "persist_dir": str(persist_dir),
            "k": k,
            "adapter_path": str(adapter_path),
        },
        "ingest_stats": ingest_stats,
        "retrieval_stats": retrieval_stats,
        "retrieved_chunks": [
            {
                "section_type": str(d.metadata.get("section_type", "other")),
                "relevance_score": float(d.metadata.get("relevance_score", 0.0) or 0.0),
                "source": str(d.metadata.get("source", "")),
                "content": (d.page_content or "").strip(),
            }
            for d in chunks
        ],
        "model_output": model_output,
        "total_time_seconds": round(time.time() - start, 3),
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run resume tuner (RAG + LoRA).")
    p.add_argument("--resume", required=True, type=Path, help="Path to resume PDF")
    p.add_argument("--job", required=True, type=Path, help="Path to job PDF")
    p.add_argument("--persist-dir", type=Path, default=Path("./tuner_db"), help="Chroma persist dir (will be reset)")
    p.add_argument("--k", type=int, default=4, help="Top-k chunks to retrieve")
    p.add_argument("--adapter", type=Path, default=Path("finetuning/resume-expert-lora"), help="LoRA adapter path")
    p.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = tune_resume(
        resume_pdf=args.resume,
        job_pdf=args.job,
        persist_dir=args.persist_dir,
        k=args.k,
        adapter_path=args.adapter,
    )

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()

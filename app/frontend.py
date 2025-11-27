import streamlit as st
import requests
import json
import os

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Career Coach", page_icon="🚀", layout="wide")

st.title("🚀 AI Agentic Resume Tuner")
st.markdown("""
This autonomous agent will:
1. **Read** the job description from your URL.
2. **Analyze** your master resume for gaps.
3. **Generate** a new, perfectly tuned PDF resume.
4. **Find** similar high-match jobs for you.
""")

# Input Section
job_url = st.text_input("Paste Job Description URL:", placeholder="https://linkedin.com/jobs/...")

if st.button("🚀 Launch Agent"):
    if not job_url:
        st.error("Please enter a valid URL.")
    else:
        with st.spinner("Agent is working... (Scraping -> RAG -> Analysis -> LaTeX -> Search)"):
            try:
                # Call the FastAPI backend
                response = requests.post(f"{API_URL}/generate-resume", json={"job_url": job_url})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Layout: 2 Columns
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("✅ Tailored Bullet Points")
                        if "tailored_bullets" in data:
                            for bullet in data["tailored_bullets"]:
                                st.success(bullet)
                        else:
                            st.warning("No specific bullets generated.")
                            
                        st.subheader("📄 Download Resume")
                        if "pdf_path" in data:
                            # extraction of filename from path might be needed depending on agent output
                            # Assuming agent returns relative path "output/tuned_resume.pdf"
                            filename = os.path.basename(data["pdf_path"])
                            download_link = f"{API_URL}/download/{filename}"
                            st.markdown(f"[**📥 Download Tuned PDF**]({download_link})")
                        else:
                            st.info("PDF generation information missing.")

                    with col2:
                        st.subheader("🔍 Similar Job Leads")
                        if "similar_jobs" in data:
                            # Sometimes Tavily returns a string representation of list, handling both
                            jobs = data["similar_jobs"]
                            if isinstance(jobs, str):
                                st.write(jobs)
                            elif isinstance(jobs, list):
                                for job in jobs:
                                    # If job is dict (from Tavily) or string
                                    if isinstance(job, dict):
                                        st.markdown(f"- [{job.get('title', 'Job Link')}]({job.get('url', '#')})")
                                    else:
                                        st.markdown(f"- {job}")
                        else:
                            st.info("No similar jobs found.")
                            
                    st.json(data) # Show raw debug data at bottom
                    
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Powered by LangChain, FastAPI, and Local LLMs.")


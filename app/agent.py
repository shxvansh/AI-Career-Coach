import os
from typing import Dict
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

# Import our custom tools
from app.tools import scrape_job_description, retrieve_resume_context, search_similar_jobs, compile_resume_pdf

# Define the Agent
def create_agent():
    # 1. Define Tools
    # We wrap our python functions into LangChain tools
    tools = [
        StructuredTool.from_function(
            func=scrape_job_description,
            name="scrape_job_description",
            description="Useful for extracting the text content from a job description URL."
        ),
        StructuredTool.from_function(
            func=retrieve_resume_context,
            name="retrieve_resume_context",
            description="Useful for retrieving relevant skills and experiences from the user's master resume based on a query."
        ),
        StructuredTool.from_function(
            func=search_similar_jobs,
            name="search_similar_jobs",
            description="Useful for finding other relevant job postings based on keywords."
        ),
        StructuredTool.from_function(
            func=compile_resume_pdf,
            name="compile_resume_pdf",
            description="Useful for generating a new PDF resume. Input should be a list of LaTeX formatted bullet points."
        )
    ]

    # 2. Initialize LLM (The Brain)
    # NOTE: In a production local setup with your Finetuned LoRA, you would use:
    # llm = HuggingFacePipeline(...) or point to a local vLLM server
    # For now, we use OpenAI for robust orchestration.
    llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

    # 3. Define the Prompt
    # This system prompt guides the agent's behavior
    system_prompt = """You are an expert Career Coach and Resume Strategist. 
    Your goal is to help the user tailor their resume for a specific job application.

    Follow this process:
    1. Scrape the provided Job Description URL to understand the requirements.
    2. Retrieve relevant experience from the user's Master Resume using the RAG tool.
    3. Analyze the gap between the Job Description and the retrieved Resume Context.
    4. Generate 3-5 high-impact, tailored bullet points in LaTeX format that bridge these gaps.
    5. Compile these new bullet points into a new PDF using the compile_resume_pdf tool.
    6. Finally, use the new keywords to search for 3 other similar jobs the user might like.

    Output the final result as a JSON object with:
    - "tailored_bullets": [list of strings],
    - "pdf_path": "path/to/resume.pdf",
    - "similar_jobs": [list of job links]
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. Create the Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # 5. Create the Executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor

def run_agent(job_url: str) -> Dict:
    agent = create_agent()
    result = agent.invoke({"input": f"Please help me apply for this job: {job_url}"})
    return result["output"]

if __name__ == "__main__":
    # Simple test
    test_url = "https://www.ycombinator.com/companies/foundry/jobs" # Example
    print("Agent initialized. Ready to run.")

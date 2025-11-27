import os
import argparse
from dotenv import load_dotenv
from app.agent import run_agent

# Load environment variables (OPENAI_API_KEY, TAVILY_API_KEY)
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="AI Career Coach - Resume Tuner Agent")
    parser.add_argument("url", help="The URL of the job description you want to apply for.")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting AI Agent for Job URL: {args.url}")
    print("--------------------------------------------------")
    
    try:
        result = run_agent(args.url)
        print("\n✅ Agent Task Completed!")
        print("--------------------------------------------------")
        print(result)
    except Exception as e:
        print(f"\n❌ Error running agent: {e}")

if __name__ == "__main__":
    main()


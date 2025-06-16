"""
Main entry point for the multi-agent system.
"""
import os
import logging
from dotenv import load_dotenv
from app.orchestration.orchestrator import Orchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("agent_activity.log", mode='w')
    ]
)

def main():
    """
    Initializes and runs the agent orchestration system.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("FATAL: OPENAI_API_KEY environment variable not set.")
        return

    model = "gpt-4o-mini"
    
    print("--- Multi-Agent System Initialized ---")
    print("Enter your request. Type 'exit' to close.")

    orchestrator = Orchestrator(api_key=api_key, model=model)

    try:
        while True:
            user_query = input("\033[94mYou > \033[0m")
            if user_query.lower() in ['exit', 'quit']:
                print("Shutting down...")
                break
            
            if not user_query.strip():
                continue

            # The orchestrator handles the entire process
            final_result = orchestrator.run(user_query)
            
            print(f"\033[92mFinal Answer >\033[0m {final_result}")

    except (KeyboardInterrupt, EOFError):
        print("\nShutting down...")

if __name__ == "__main__":
    main() 
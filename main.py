import logging
import os
from dotenv import load_dotenv
from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    edit_file_tool, edit_file_tool_def,
    list_files_tool, list_files_tool_def,
    run_tests_tool, run_tests_tool_def,
)
from app.orchestration.decomposer import TaskDecomposer
from app.orchestration.orchestrator import Orchestrator
from app.agents.roles import (
    CodingAgent,
    ReviewerAgent,
    TestingAgent,
    EvaluatorAgent,
)

def main():
    """Main function to run the AI agent."""
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        handlers=[
            logging.FileHandler("agent_activity.log"),
            logging.StreamHandler()
        ]
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        print("Error: Please make sure your .env file contains OPENAI_API_KEY.")
        return

    try:
        # 1. Initialize the agent team
        logging.info("Initializing agent team...")
        
        common_kwargs = {"api_key": api_key, "model": "o4-mini"}

        coding_agent = CodingAgent(
            name="CodingAgent", 
            use_rag=True, 
            rag_config={
                "top_k": 3,
                "filters": {"tags": ["code-example", "style-guide"]}
            },
            **common_kwargs
        )
        coding_agent.add_tool(read_file_tool, read_file_tool_def)
        coding_agent.add_tool(edit_file_tool, edit_file_tool_def)
        coding_agent.add_tool(list_files_tool, list_files_tool_def)

        testing_agent = TestingAgent(
            name="TestingAgent", 
            use_rag=True, 
            rag_config={
                "top_k": 2,
                "filters": {"tags": ["testing-guide"]}
            },
            **common_kwargs
        )
        testing_agent.add_tool(read_file_tool, read_file_tool_def)
        testing_agent.add_tool(run_tests_tool, run_tests_tool_def)

        evaluator_agent = EvaluatorAgent(
            name="EvaluatorAgent", 
            use_rag=True, 
            rag_config={
                "top_k": 4,
                "filters": {"tags": ["error-analysis", "debugging"]}
            },
            **common_kwargs
        )
        evaluator_agent.add_tool(read_file_tool, read_file_tool_def)

        reviewer_agent = ReviewerAgent(
            name="ReviewerAgent",
            # This agent already has use_rag=True in its own __init__
            rag_config={
                "top_k": 5,
                "filters": {"tags": ["style-guide"]}
            },
            **common_kwargs
        )
        reviewer_agent.add_tool(read_file_tool, read_file_tool_def)

        workers = {
            "CodingAgent": coding_agent,
            "TestingAgent": testing_agent,
            "EvaluatorAgent": evaluator_agent,
            "ReviewerAgent": reviewer_agent,
        }
        
        default_agent = Agent(
            name="DefaultAgent",
            role="General Assistant",
            goal="Perform basic tasks like listing files when no other agent is assigned.",
            **common_kwargs
        )
        default_agent.add_tool(list_files_tool, list_files_tool_def)
        default_agent.add_tool(read_file_tool, read_file_tool_def)
        workers["DefaultAgent"] = default_agent

        # Planner
        planner = TaskDecomposer(api_key=api_key, model="o4-mini")
        
        # The orchestrator now manages the team
        orchestrator = Orchestrator(
            task_decomposer=planner,
            worker_agents=workers
        )

        # 2. Request the goal and run the orchestrator
        goal = input("Please enter your high-level goal: ")
        if not goal:
            print("Goal cannot be empty.")
            return

        orchestrator.run(goal)

    except Exception as e:
        logging.critical("A critical error occurred: %s", e, exc_info=True)
        print(f"\nCritical Error: {e}")

if __name__ == "__main__":
    main() 
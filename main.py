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

        coding_agent = CodingAgent(name="CodingAgent", **common_kwargs)
        coding_agent.add_tool(read_file_tool, read_file_tool_def)
        coding_agent.add_tool(edit_file_tool, edit_file_tool_def)
        coding_agent.add_tool(list_files_tool, list_files_tool_def)

        testing_agent = TestingAgent(name="TestingAgent", **common_kwargs)
        testing_agent.add_tool(read_file_tool, read_file_tool_def)
        testing_agent.add_tool(run_tests_tool, run_tests_tool_def)

        evaluator_agent = EvaluatorAgent(name="EvaluatorAgent", **common_kwargs)
        evaluator_agent.add_tool(read_file_tool, read_file_tool_def)

        reviewer_agent = ReviewerAgent(name="ReviewerAgent", **common_kwargs)
        reviewer_agent.add_tool(read_file_tool, read_file_tool_def)

        workers = {
            "CodingAgent": coding_agent,
            "TestingAgent": testing_agent,
            "EvaluatorAgent": evaluator_agent,
            "ReviewerAgent": reviewer_agent,
        }
        
        default_agent = Agent(name="DefaultAgent", **common_kwargs)
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
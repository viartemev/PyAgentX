import logging
import os
import yaml
from importlib import import_module
from dotenv import load_dotenv
from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    edit_file_tool, edit_file_tool_def,
    list_files_tool, list_files_tool_def,
    run_tests_tool, run_tests_tool_def,
)
from app.orchestration.orchestrator import Orchestrator

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_class_from_string(class_path: str):
    """Dynamically imports a class from a string path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def main():
    """Main function to run the AI agent."""
    load_dotenv()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        handlers=[
            logging.FileHandler("agent_activity.log", mode='w'),
            logging.StreamHandler()
        ]
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not found in .env file.")
        print("Error: Please make sure your .env file contains OPENAI_API_KEY.")
        return

    try:
        logging.info("Loading configurations...")
        main_config = load_config('configs/config.yaml')
        
        workers = {}
        common_kwargs = {"api_key": api_key, "model": main_config.get('default_model', 'o4-mini')}

        for agent_name, agent_info in main_config['agents'].items():
            logging.info(f"Initializing {agent_name}...")
            
            # Load agent-specific config
            agent_config_path = agent_info['config_path']
            agent_config = load_config(agent_config_path)

            # Get the agent class dynamically
            agent_class = get_class_from_string(agent_info['_target_'])
            
            # Prepare initialization parameters
            init_params = agent_config.copy()
            init_params.update(common_kwargs)
            
            # Create agent instance
            agent_instance = agent_class(**init_params)

            # Add tools based on agent name/role
            if agent_name == "CodingAgent":
                agent_instance.add_tool(read_file_tool, read_file_tool_def)
                agent_instance.add_tool(edit_file_tool, edit_file_tool_def)
                agent_instance.add_tool(list_files_tool, list_files_tool_def)
            elif agent_name == "TestingAgent":
                agent_instance.add_tool(read_file_tool, read_file_tool_def)
                agent_instance.add_tool(run_tests_tool, run_tests_tool_def)
            elif agent_name in ["ReviewerAgent", "EvaluatorAgent", "DefaultAgent"]:
                agent_instance.add_tool(read_file_tool, read_file_tool_def)
            
            workers[agent_name] = agent_instance

        # The orchestrator now manages the team
        orchestrator = Orchestrator(
            task_decomposer=workers.pop("TaskDecomposer"),
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
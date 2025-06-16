"""
Factory for creating agent teams from configuration.
"""
import logging
import yaml
import os
from importlib import import_module
from typing import Dict, Tuple, Any, List

from app.agents.agent import Agent
from app.agents.roles.task_decomposer import TaskDecomposer
from app.agents.tools import read_file_tool, read_file_tool_def, list_files_tool, list_files_tool_def
from app.agents.web_search_tool import web_search_tool, web_search_tool_def
from app.agents.memory_tool import save_memory_tool, save_memory_tool_def

# A mapping of tool names to their functions and definitions
AVAILABLE_TOOLS = {
    "read_file": (read_file_tool, read_file_tool_def),
    "list_files": (list_files_tool, list_files_tool_def),
    "web_search": (web_search_tool, web_search_tool_def),
    "save_memory": (save_memory_tool, save_memory_tool_def),
}

def load_config(path: str) -> dict:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_class_from_string(class_path: str):
    """Dynamically imports a class from a string path."""
    module_path, class_name = class_path.rsplit('.', 1)
    module = import_module(module_path)
    return getattr(module, class_name)

def create_agent_team(main_config_path: str) -> Tuple[TaskDecomposer, Dict[str, Agent]]:
    """
    Creates and configures the agent team based on YAML files.

    Args:
        main_config_path (str): The path to the main configuration file.

    Returns:
        A tuple containing the TaskDecomposer and the dictionary of worker agents.
    """
    logging.info("Loading configurations from %s...", main_config_path)
    main_config = load_config(main_config_path)
    
    workers = {}
    api_key = os.getenv("OPENAI_API_KEY")
    common_kwargs = {"api_key": api_key, "model": main_config.get('default_model', 'o4-mini')}

    for agent_name, agent_info in main_config['agents'].items():
        logging.info(f"Initializing {agent_name}...")
        
        agent_config_path = agent_info['config_path']
        agent_config = load_config(agent_config_path)

        agent_class = get_class_from_string(agent_info['_target_'])
        
        init_params = agent_config.copy()
        init_params.update(common_kwargs)
        
        agent_instance = agent_class(**init_params)

        # The logic for adding tools has been moved to the agent classes themselves.
        # The factory is now cleaner and only responsible for instantiation.
        
        workers[agent_name] = agent_instance

    task_decomposer = workers.pop("TaskDecomposer")
    return task_decomposer, workers 

class AgentFactory:
    """
    A factory class for creating different types of specialized agents.
    """
    @staticmethod
    def create_agent(
        agent_config: Dict[str, Any],
        api_key: str,
        model: str
    ) -> Agent:
        """
        Creates an agent based on a configuration dictionary.

        Args:
            agent_config: A dictionary containing the agent's name, role, goal,
                          and a list of tool names it should have.
            api_key: The OpenAI API key.
            model: The name of the model to use.

        Returns:
            An instance of the Agent class, configured as specified.
        """
        name = agent_config.get("name", "SpecializedAgent")
        role = agent_config.get("role", "An assistant")
        goal = agent_config.get("goal", "To complete tasks efficiently.")
        tool_names = agent_config.get("tools", [])

        agent = Agent(
            name=name,
            role=role,
            goal=goal,
            api_key=api_key,
            model=model
        )

        # Clear default tools and add only the specified ones
        agent.tools = {}
        agent.tool_definitions = []
        
        for tool_name in tool_names:
            if tool_name in AVAILABLE_TOOLS:
                tool_func, tool_def = AVAILABLE_TOOLS[tool_name]
                agent.add_tool(tool_func, tool_def)
        
        return agent 
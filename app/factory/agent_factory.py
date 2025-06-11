"""
Factory for creating agent teams from configuration.
"""
import logging
import yaml
import os
from importlib import import_module
from typing import Dict, Tuple

from app.agents.agent import Agent
from app.agents.roles.task_decomposer import TaskDecomposer

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
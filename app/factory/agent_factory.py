"""
Factory for creating specialized agents.
"""
from typing import Dict, Any

from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    list_files_tool, list_files_tool_def,
    write_to_file_tool, write_to_file_tool_def,
    run_tests_tool, run_tests_tool_def
)
from app.agents.web_search_tool import web_search_tool, web_search_tool_def

# A mapping of tool names to their functions and definitions
# This makes it easy to add tools to agents based on their config.
AVAILABLE_TOOLS = {
    "read_file": (read_file_tool, read_file_tool_def),
    "list_files": (list_files_tool, list_files_tool_def),
    "write_to_file": (write_to_file_tool, write_to_file_tool_def),
    "run_tests": (run_tests_tool, run_tests_tool_def),
    "web_search": (web_search_tool, web_search_tool_def),
}

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
            agent_config: A dictionary containing the agent's name, role, and goal.
            api_key: The OpenAI API key.
            model: The name of the model to use.

        Returns:
            An instance of the Agent class, configured as specified.
        """
        name = agent_config.get("name", "SpecializedAgent")
        role = agent_config.get("role", "An assistant")
        goal = agent_config.get("goal", "To complete tasks efficiently.")

        # In our new simplified system, the base Agent class already
        # includes all the necessary tools by default.
        # We just need to instantiate it with the correct persona.
        agent = Agent(
            name=name,
            role=role,
            goal=goal,
            api_key=api_key,
            model=model
        )
        
        return agent 
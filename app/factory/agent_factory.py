"""
Factory for creating specialized agents.
"""
from typing import Dict, Any

from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    list_files_tool, list_files_tool_def,
    write_to_file_tool, write_to_file_tool_def,
    update_file_tool, update_file_tool_def,
    run_tests_tool, run_tests_tool_def
)
from app.agents.web_search_tool import web_search_tool, web_search_tool_def
from app.agents.memory_tool import save_memory_tool, save_memory_tool_def

# A mapping of tool names to their functions and definitions
# This makes it easy to add tools to agents based on their config.
AVAILABLE_TOOLS = {
    "read_file": (read_file_tool, read_file_tool_def),
    "list_files": (list_files_tool, list_files_tool_def),
    "write_to_file": (write_to_file_tool, write_to_file_tool_def),
    "update_file": (update_file_tool, update_file_tool_def),
    "run_tests": (run_tests_tool, run_tests_tool_def),
    "web_search": (web_search_tool, web_search_tool_def),
    "save_memory": (save_memory_tool, save_memory_tool_def),
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
        tool_names = agent_config.get("tools", [])

        # Create a base agent
        agent = Agent(
            name=name,
            role=role,
            goal=goal,
            api_key=api_key,
            model=model
        )

        # Add only the tools specified in the agent's configuration
        for tool_name in tool_names:
            if tool_name in AVAILABLE_TOOLS:
                tool_func, tool_def = AVAILABLE_TOOLS[tool_name]
                agent.add_tool(tool_func, tool_def)
            else:
                # This could be a configuration error, but we'll just log it for now.
                print(f"Warning: Tool '{tool_name}' not found for agent '{name}'.")

        # The 'save_memory' tool is essential for all agents to learn.
        agent.add_tool(save_memory_tool, save_memory_tool_def)
        
        return agent 
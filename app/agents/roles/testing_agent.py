"""Testing Agent."""
from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    run_tests_tool, run_tests_tool_def,
)

class TestingAgent(Agent):
    """An agent specializing in running tests."""
    def __init__(self, name: str, role: str, goal: str, **kwargs):
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            **kwargs,
        )
        # Self-register tools
        self.add_tool(read_file_tool, read_file_tool_def)
        self.add_tool(run_tests_tool, run_tests_tool_def)

        # The system_prompt is now handled by the base Agent class. 
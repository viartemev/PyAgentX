"""Reviewer Agent."""
from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
)

class ReviewerAgent(Agent):
    """An agent specializing in strict Code Review."""
    def __init__(self, name: str, role: str, goal: str, **kwargs):
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            **kwargs,
        )
        # Self-register tools
        self.add_tool(read_file_tool, read_file_tool_def)

        self.system_prompt = (
            "You are a Senior Software Engineer acting as a code reviewer. "
            "Your task is to provide a thorough review of the code based on the "
            "provided file path. Use your available tools to read the file content.\n\n"
            "If the code meets all standards, respond with only the word 'LGTM'.\n"
            "Otherwise, provide clear, constructive feedback on what needs to be improved."
        )
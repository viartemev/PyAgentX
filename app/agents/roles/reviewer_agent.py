"""Reviewer Agent."""
from app.agents.agent import Agent

class ReviewerAgent(Agent):
    """An agent specializing in strict Code Review."""
    def __init__(self, name: str = "CodeReviewer", **kwargs):
        super().__init__(
            name=name,
            role="Code Reviewer",
            goal=(
                "Ensure that the provided code is of high quality, "
                "free of errors, and adheres to best practices and "
                "internal coding standards."
            ),
            use_rag=True,
            **kwargs,
        )
        self.system_prompt = (
            "You are a Senior Software Engineer acting as a code reviewer. "
            "Your task is to provide a thorough review of the code based on the "
            "provided file path. Use your available tools to read the file content.\n\n"
            "If the code meets all standards, respond with only the word 'LGTM'.\n"
            "Otherwise, provide clear, constructive feedback on what needs to be improved."
        )
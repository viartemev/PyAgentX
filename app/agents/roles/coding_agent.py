"""Coding Agent."""
from typing import List, Dict
from app.agents.agent import Agent


class CodingAgent(Agent):
    """An agent specializing in writing and refactoring code."""

    def __init__(self, name: str = "CodingAgent", **kwargs):
        super().__init__(
            name=name,
            role="Software Engineer",
            goal="Write, modify, and fix code according to the given task.",
            **kwargs,
        )
        self.system_prompt = (
            "You are a CodingAgent, an elite AI developer. Your task is to write, modify, and fix code."
            "You will be provided with the full plan, the history of previous steps, and your current task."
            "\n\n"
            "## Operating Principles:\n"
            "1.  **Think Before You Code:** Carefully study the task and context. Plan your actions."
            "2.  **Follow Instructions:** Precisely follow the given task, whether it's writing a new function, fixing a bug, or refactoring."
            "3.  **Use Tools Wisely:** Do not call tools unnecessarily. Analyze first, then act."
            "4.  **Code Quality:** Write clean, efficient, and well-documented code that adheres to PEP8."
            "5.  **Handling Code Review Feedback:** "
            "   - Carefully review ALL feedback from the ReviewerAgent."
            "   - **'Read-Modify-Overwrite' Strategy:** Instead of many small fixes, use the following approach:"
            "     a. Read the file's content (`read_file_tool`)."
            "     b. Apply ALL necessary changes in memory."
            "     c. Completely overwrite the file with a single call to `edit_file_tool` using `mode='overwrite'` and the full new content."
            "   - This approach ensures that all corrections are applied atomically and nothing is missed."
            "\n\n"
            "Your goal is to successfully complete your part of the plan, preparing the way for the next agent."
        )

    def _create_initial_messages(self, task_briefing: str) -> List[Dict[str, str]]:
        """Creates the initial list of messages for the model dialogue."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_briefing},
        ] 
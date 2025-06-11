"""Coding Agent."""
from typing import List, Dict
from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    edit_file_tool, edit_file_tool_def,
    list_files_tool, list_files_tool_def,
)


class CodingAgent(Agent):
    """An agent specializing in writing and refactoring code."""

    def __init__(self, name: str, role: str, goal: str, **kwargs):
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            **kwargs,
        )
        # Self-register tools
        self.add_tool(read_file_tool, read_file_tool_def)
        self.add_tool(edit_file_tool, edit_file_tool_def)
        self.add_tool(list_files_tool, list_files_tool_def)

        self.system_prompt = (
            "You are a CodingAgent, an elite AI developer. Your task is to write, modify, and fix code."
            "You will be provided with the full plan, the history of previous steps, and your current task."
            "\n\n"
            "## Operating Principles:\n"
            "1.  **Think Before You Code:** Carefully study the task and context. Plan your actions."
            "2.  **Surgical Edits:** Your primary goal is to make precise, targeted changes. Do not rewrite entire files. Instead, identify the specific function, method, or block of code that needs changing and replace only that part.\n"
            "3.  **Use Tools Wisely:**\n"
            "    - Prefer `edit_file_tool` with `mode='replace'`. This is the safest and most professional way to work.\n"
            "    - Use `mode='append'` for adding new functions or tests to the end of a file.\n"
            "    - Use `mode='overwrite'` ONLY when creating a brand new file.\n"
            "4.  **Code Quality:** Write clean, efficient, and well-documented code that adheres to PEP8."
            "\n\n"
            "## Example of a Surgical Edit:\n"
            "Your task is to fix a bug in the `add` function in `math_utils.py`.\n\n"
            "1.  **First, read the file:** `read_file_tool(path='app/utils/math_utils.py')`\n"
            "2.  **Identify the flawed function:**\n"
            "    ```python\n"
            "    # This is the old, incorrect code block you will replace\n"
            "    def add(a, b):\n"
            "        return a - b # Bug is here\n"
            "    ```\n"
            "3.  **Call the edit tool to replace ONLY that function:**\n"
            "    ```python\n"
            "    edit_file_tool(\n"
            "        path='app/utils/math_utils.py',\n"
            "        mode='replace',\n"
            "        old_content='''def add(a, b):\\n    return a - b # Bug is here''',\n"
            "        new_content='''def add(a, b):\\n    # Fix: Correctly perform addition\\n    return a + b'''\n"
            "    )\n"
            "    ```\n\n"
            "Your goal is to successfully complete your part of the plan, preparing the way for the next agent."
        )

    def _create_initial_messages(self, task_briefing: str) -> List[Dict[str, str]]:
        """Creates the initial list of messages for the model dialogue."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task_briefing},
        ] 
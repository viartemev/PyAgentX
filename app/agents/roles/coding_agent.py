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
        # Добавляем специфику роли в цель, чтобы она попала в основной промпт
        refined_goal = (
            f"{goal}\n\n"
            "IMPORTANT: Your primary goal is to make precise, targeted changes (surgical edits). "
            "Do not rewrite entire files. Instead, identify the specific function, method, "
            "or block of code that needs changing and use the 'edit_file_tool' to replace only that part."
        )

        super().__init__(
            name=name,
            role=role,
            goal=refined_goal, # Передаем уточненную цель
            **kwargs,
        )
        # Self-register tools
        self.add_tool(read_file_tool, read_file_tool_def)
        self.add_tool(edit_file_tool, edit_file_tool_def)
        self.add_tool(list_files_tool, list_files_tool_def)

        # Старый system_prompt и _create_initial_messages больше не нужны,
        # так как вся логика теперь в базовом классе Agent. 
"""Coding Agent."""
from typing import List, Dict
from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    write_to_file_tool, write_to_file_tool_def,
    list_files_tool, list_files_tool_def,
    run_tests_tool, run_tests_tool_def
)


class CodingAgent(Agent):
    """An agent specializing in writing and refactoring code."""

    def __init__(self, name: str, role: str, goal: str, **kwargs):
        # Добавляем специфику роли в цель, чтобы она попала в основной промпт
        refined_goal = (
            f"{goal}\n\n"
            "IMPORTANT: Your primary goal is to write high-quality, efficient, and clean Python code. "
            "You must follow the provided coding standards. "
            "When you need to modify a file, read its content first, then provide the full, complete, updated content to the 'write_to_file_tool'. "
            "This tool will overwrite the entire file with your new content."
        )

        super().__init__(
            name=name,
            role=role,
            goal=refined_goal, # Передаем уточненную цель
            **kwargs,
        )
        # Self-register tools
        self.add_tool(read_file_tool, read_file_tool_def)
        self.add_tool(list_files_tool, list_files_tool_def)
        self.add_tool(write_to_file_tool, write_to_file_tool_def)
        self.add_tool(run_tests_tool, run_tests_tool_def)

        # Старый system_prompt и _create_initial_messages больше не нужны,
        # так как вся логика теперь в базовом классе Agent. 
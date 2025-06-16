"""
This module defines the memory tool for the agent.
"""
from typing import Dict, Any
from app.memory.memory_manager import MemoryManager

# We will instantiate the manager when the agent loads the tool.
# This avoids creating a new connection for every call.
memory_manager = MemoryManager()

# 1. Tool Definition (JSON Schema)
save_memory_tool_def = {
    "type": "function",
    "function": {
        "name": "save_memory_tool",
        "description": "Saves a specific piece of information to your long-term memory for future use. "
                       "Use this when you learn a new, important fact that needs to be remembered across sessions.",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The specific fact or piece of information to save."
                }
            },
            "required": ["fact"],
        },
    },
}

# 2. Tool Function
def save_memory_tool(args: Dict[str, Any]) -> str:
    """
    Saves a fact to the long-term memory.

    Args:
        args: A dictionary containing the 'fact' to be saved.

    Returns:
        A confirmation string of the save operation.
    """
    fact = args.get("fact")
    if not fact:
        return "Error: The 'fact' parameter is required to save to memory."
    
    result = memory_manager.add_fact(fact)
    return result.get("message", "An unknown error occurred.") 
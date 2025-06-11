### Agent Tool Creation Guide

This document outlines the best practices for creating new tool functions that can be used by AI agents in our system.

#### 1. Tool Function Structure

Every tool must be a wrapper around the core logic and accept a **single argument** of type `Dict[str, Any]`. This ensures a unified interface for all tools.

```python
# Correct
def my_tool(input_data: Dict[str, Any]) -> str:
    #...
    
# Incorrect
def my_tool(param1: str, param2: int) -> str:
    #...
```

#### 2. Mandatory Error Handling

A tool should **never crash with an unhandled exception**. Always use `try-except` and return an informative error message as a string.

```python
def substring_tool(input_data: Dict[str, Any]) -> str:
    try:
        text = input_data['text']
        start = input_data['start']
        # ... core logic ...
        return result
    except KeyError as e:
        return f"Error: Missing required key {e} in input_data."
    except Exception as e:
        return f"An unexpected error occurred: {e}"
```

#### 3. Detailed Docstrings

Always write detailed docstrings in Google-style. Describe the function's purpose, all keys in the `input_data` dictionary, and the return value.

#### 4. Tool Definition (`_tool_def`)

Every tool must have a corresponding `_tool_def` definition. This is a dictionary that describes the function's signature for the OpenAI API, allowing the agent to understand how to call your tool.

```python
substring_tool_def = {
    "type": "function",
    "function": {
        "name": "substring_tool",
        "description": "Extracts a substring from text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The source text."},
                "start": {"type": "integer", "description": "The starting index."},
            },
            "required": ["text", "start"],
        },
    },
}
``` 
from typing import Callable, Any, Dict

class ToolDefinition:
    def __init__(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        function: Callable[[Dict[str, Any]], str],
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.function = function

def read_file_tool(input_data: Dict[str, Any]) -> str:
    path = input_data.get("path")
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

read_file_definition = ToolDefinition(
    name="read_file",
    description="Read the contents of a given relative file path.",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "The relative file path."}},
        "required": ["path"],
    },
    function=read_file_tool,
) 
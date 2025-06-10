from typing import Callable, Any, Dict

class ToolDefinition:
    """Определяет структуру инструмента, его описание и функцию."""
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

    def to_openai_spec(self) -> Dict[str, Any]:
        """Преобразует определение инструмента в формат, ожидаемый OpenAI API."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }

def read_file_tool(input_data: Dict[str, Any]) -> str:
    """
    Читает содержимое файла по указанному пути.

    Args:
        input_data (Dict[str, Any]): Словарь, содержащий ключ 'path' с путём к файлу.

    Returns:
        Содержимое файла в виде строки или сообщение об ошибке.
    """
    path = input_data.get("path")
    if not path:
        return "Ошибка: Аргумент 'path' обязателен."
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Ошибка: Файл не найден по пути '{path}'."
    except Exception as e:
        return f"Ошибка: Не удалось прочитать файл '{path}': {e}"

read_file_definition = ToolDefinition(
    name="read_file",
    description="Читает содержимое файла по указанному пути. Используйте, чтобы посмотреть, что внутри файла.",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "Относительный или абсолютный путь к файлу."}},
        "required": ["path"],
    },
    function=read_file_tool,
) 
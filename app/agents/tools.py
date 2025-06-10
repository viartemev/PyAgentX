"""
Этот модуль определяет инструменты (tools), которые может использовать AI-агент.
Каждый инструмент представлен классом ToolDefinition и соответствующей функцией.
"""
from typing import Callable, Any, Dict
import os
import logging

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
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
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
    except IOError as e:
        logging.error("Ошибка ввода-вывода при чтении файла %s: %s", path, e)
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

def list_files_tool(input_data: Dict[str, Any]) -> str:
    """
    Рекурсивно выводит список файлов и директорий по указанному пути.

    Args:
        input_data (Dict[str, Any]): Словарь, который может содержать ключ 'path'.
                                      Если 'path' не указан, используется текущая директория.

    Returns:
        Отформатированный список содержимого директории или сообщение об ошибке.
    """
    path = input_data.get("path", ".")
    try:
        if not os.path.isdir(path):
            return f"Ошибка: Путь '{path}' не является директорией или не существует."

        output = f"Содержимое директории '{os.path.abspath(path)}':\n"
        for root, dirs, files in os.walk(path):
            # Исключаем служебные директории
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.venv', '.idea', '.git']]
            
            level = root.replace(path, '').count(os.sep)
            indent = ' ' * 4 * (level)
            if level > 0:
                output += f'{indent}{os.path.basename(root)}/\n'
            
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                output += f'{sub_indent}{f}\n'
        return output.strip()
    except OSError as e:
        logging.error("Ошибка ОС при листинге файлов в %s: %s", path, e)
        return f"Ошибка: Не удалось получить список файлов для '{path}': {e}"

list_files_definition = ToolDefinition(
    name="list_files",
    description="Выводит список файлов и директорий по указанному пути, чтобы понять структуру проекта. По умолчанию показывает содержимое текущей директории.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Путь к директории. Например, '.' для текущей или 'app/agents' для вложенной.",
            }
        },
    },
    function=list_files_tool,
)

def edit_file_tool(input_data: Dict[str, Any]) -> str:
    """
    Перезаписывает файл по указанному пути новым содержимым.

    Args:
        input_data (Dict[str, Any]): Словарь, содержащий 'path' и 'content'.

    Returns:
        Сообщение об успехе или ошибке.
    """
    path = input_data.get("path")
    content = input_data.get("content")

    if not path or content is None:
        return "Ошибка: Аргументы 'path' и 'content' обязательны."
    
    try:
        # Убедимся, что директория для файла существует
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Файл '{path}' успешно сохранен."
    except IOError as e:
        logging.error("Ошибка ввода-вывода при записи в файл %s: %s", path, e)
        return f"Ошибка: Не удалось записать в файл '{path}': {e}"

edit_file_definition = ToolDefinition(
    name="edit_file",
    description="Создает или полностью перезаписывает файл по указанному пути указанным содержимым. Используйте с осторожностью, так как старое содержимое будет удалено.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Путь к файлу, который нужно создать или перезаписать.",
            },
            "content": {
                "type": "string",
                "description": "Новое содержимое файла.",
            }
        },
        "required": ["path", "content"],
    },
    function=edit_file_tool,
)

def delete_file_tool(input_data: Dict[str, Any]) -> str:
    """
    Удаляет файл по указанному пути.

    Args:
        input_data (Dict[str, Any]): Словарь, содержащий 'path' к файлу.

    Returns:
        Сообщение об успехе или ошибке.
    """
    path = input_data.get("path")
    if not path:
        return "Ошибка: Аргумент 'path' обязателен."

    try:
        if not os.path.isfile(path):
            return f"Ошибка: Файл по пути '{path}' не найден и не может быть удален."
        
        os.remove(path)
        return f"Файл '{path}' успешно удален."
    except OSError as e:
        logging.error("Ошибка ОС при удалении файла %s: %s", path, e)
        return f"Ошибка: Не удалось удалить файл '{path}': {e}"

delete_file_definition = ToolDefinition(
    name="delete_file",
    description="Удаляет файл по указанному пути. Это действие необратимо. Используйте с большой осторожностью.",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Путь к файлу, который нужно удалить.",
            }
        },
        "required": ["path"],
    },
    function=delete_file_tool,
) 
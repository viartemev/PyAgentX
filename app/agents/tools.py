"""
Этот модуль определяет инструменты (tools), которые может использовать AI-агент.
Каждый инструмент представлен классом ToolDefinition и соответствующей функцией.
"""
from typing import Callable, Any, Dict, Union
from numbers import Real
import os
import logging
import subprocess
import sys


def read_file_tool(input_data: Dict[str, Any]) -> str:
    """
    Читает содержимое файла по указанному пути.

    Args:
        input_data (Dict[str, Any]): Словарь, содержащий ключ 'path' с путём к файлу.

    Returns:
        Содержимое файла в виде строки или сообщение об ошибке.
    """
    try:
        path = input_data["path"]
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Ошибка: Не удалось прочитать файл '{input_data.get('path')}': {e}"


def list_files_tool(input_data: Dict[str, Any]) -> str:
    """
    Выводит список файлов и директорий по указанному пути.

    Args:
        input_data (Dict[str, Any]): Словарь, который может содержать ключ 'path'
                                      с путем к директории. По умолчанию - текущая.
    Returns:
        Отформатированное дерево файлов и директорий в виде строки.
    """
    path = input_data.get("path", ".")
    if not os.path.isdir(path):
        return f"Ошибка: Путь '{path}' не является директорией или не существует."

    ignore_dirs = {'.git', '.idea', '.venv', '__pycache__', '.pytest_cache'}
    output = ""

    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        if root != path:
            output += f"{indent}{os.path.basename(root)}/\n"
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            output += f"{sub_indent}{f}\n"
    return output.strip()


def write_to_file_tool(input_data: Dict[str, Any]) -> str:
    """
    Создает новый файл или полностью перезаписывает существующий.

    Args:
        input_data (Dict[str, Any]): Словарь, содержащий:
            'path' (str): Путь к файлу.
            'content' (str): Содержимое для записи.
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
        return f"Файл '{path}' успешно создан/перезаписан."
    except Exception as e:
        return f"Ошибка: Не удалось записать в файл '{path}': {e}"


def delete_file_tool(input_data: Dict[str, Any]) -> str:
    """
    Удаляет файл по указанному пути.

    Args:
        input_data (Dict[str, Any]): Словарь, содержащий 'path' к файлу.

    Returns:
        Сообщение об успехе или ошибке.
    """
    try:
        path = input_data["path"]
        os.remove(path)
        return f"Файл '{path}' успешно удален."
    except Exception as e:
        return f"Ошибка: Не удалось удалить файл '{input_data.get('path')}': {e}"


def run_tests_tool(input_data: Dict[str, Any]) -> str:
    """
    Запускает тесты с помощью pytest для указанного файла или директории и возвращает результат.
    Используй этот инструмент, чтобы проверить корректность кода после его написания или модификации.

    Args:
        input_data (Dict[str, Any]): Словарь, который может содержать ключ 'path'
                                      с путем к тестовому файлу или директории.
                                      Если 'path' не указан, pytest будет запущен для всего проекта.

    Returns:
        Результат выполнения pytest (stdout + stderr) в виде строки.
    """
    path = input_data.get("path", ".")
    logging.info("Запуск pytest для пути: %s", path)
    try:
        command = [sys.executable, "-m", "pytest", path]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120
        )
        output = f"Pytest stdout:\n{result.stdout}\n"
        if result.stderr:
            output += f"Pytest stderr:\n{result.stderr}\n"
        if result.returncode == 0:
            return f"УСПЕХ: Все тесты пройдены.\n\n{output}"
        elif result.returncode == 1:
            return f"ПРОВАЛ: Некоторые тесты не прошли.\n\n{output}"
        elif result.returncode == 5:
            return f"ПРЕДУПРЕЖДЕНИЕ: Pytest не нашел тесты для запуска по пути '{path}'.\n\n{output}"
        else:
            return f"ОШИБКА: Pytest завершился с кодом {result.returncode}.\n\n{output}"
    except FileNotFoundError:
        return "Критическая ошибка: команда 'pytest' не найдена."
    except subprocess.TimeoutExpired:
        return "Ошибка: Выполнение тестов заняло слишком много времени и было прервано."
    except Exception as e:
        logging.error("Непредвиденная ошибка при запуске pytest: %s", e, exc_info=True)
        return f"Критическая ошибка: Не удалось запустить тесты. Причина: {e}"


def update_file_tool(input_data: Dict[str, Any]) -> str:
    """
    Appends content to an existing file. If the file doesn't exist, it creates it.

    Args:
        input_data (Dict[str, Any]): A dictionary containing:
            'path' (str): The path to the file.
            'content' (str): The content to append to the file.
    """
    path = input_data.get("path")
    content = input_data.get("content")

    if not path or content is None:
        return "Error: 'path' and 'content' are required arguments."

    try:
        # 'a' mode appends to the file, and creates it if it doesn't exist.
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n\n" + content)
        return f"Content successfully appended to '{path}'."
    except Exception as e:
        return f"Error: Could not update file '{path}': {e}"


# Определения инструментов (Tool Definitions)
read_file_tool_def = {
    "type": "function",
    "function": {
        "name": "read_file_tool",
        "description": "Читает содержимое файла по указанному пути.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Полный путь к файлу для чтения."}
            },
            "required": ["path"],
        },
    },
}

list_files_tool_def = {
    "type": "function",
    "function": {
        "name": "list_files_tool",
        "description": "Рекурсивно выводит дерево файлов и директории по указанному пути.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к директории для просмотра. По умолчанию '.'"}
            },
            "required": [],
        },
    },
}

write_to_file_tool_def = {
    "type": "function",
    "function": {
        "name": "write_to_file_tool",
        "description": "Создает новый файл или полностью перезаписывает существующий указанным контентом.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Полный путь к файлу (включая имя файла)."},
                "content": {"type": "string", "description": "Полное содержимое для записи в файл."},
            },
            "required": ["path", "content"],
        },
    },
}

delete_file_tool_def = {
    "type": "function",
    "function": {
        "name": "delete_file_tool",
        "description": "Удаляет файл по указанному пути.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Полный путь к файлу для удаления."}
            },
            "required": ["path"],
        },
    },
}

run_tests_tool_def = {
    "type": "function",
    "function": {
        "name": "run_tests_tool",
        "description": "Запускает тесты с помощью pytest для указанного файла или директории.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к тестовому файлу или директории. По умолчанию - весь проект."}
            },
            "required": [],
        },
    },
}

update_file_tool_def = {
    "type": "function",
    "function": {
        "name": "update_file_tool",
        "description": "Appends content to an existing file. Creates the file if it does not exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to be updated."
                },
                "content": {
                    "type": "string",
                    "description": "The content to append to the file."
                }
            },
            "required": ["path", "content"]
        }
    }
}

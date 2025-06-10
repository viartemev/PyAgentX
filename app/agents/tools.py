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
    Рекурсивно выводит дерево файлов и директорий по указанному пути,
    игнорируя служебные файлы/директории. Помогает понять структуру проекта.
    
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


def edit_file_tool(input_data: Dict[str, Any]) -> str:
    """
    Создает, перезаписывает, добавляет или заменяет контент в файле.
    - 'overwrite': Полностью перезаписывает файл.
    - 'append': Добавляет контент в конец файла.
    - 'replace': Заменяет один фрагмент строки на другой.

    Args:
        input_data (Dict[str, Any]): Словарь, содержащий:
            'path' (str): Путь к файлу.
            'mode' (str): Режим работы ('overwrite', 'append', 'replace').
            'content' (str, optional): Содержимое для 'overwrite' или 'append'.
            'old_content' (str, optional): Исходный фрагмент для 'replace'.
            'new_content' (str, optional): Новый фрагмент для 'replace'.
    """
    path = input_data.get("path")
    mode = input_data.get("mode", "overwrite")

    if not path:
        return "Ошибка: Аргумент 'path' обязателен."
    if mode not in ['overwrite', 'append', 'replace']:
        return "Ошибка: Недопустимый режим. Используйте 'overwrite', 'append' или 'replace'."

    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        if mode == 'replace':
            old_content = input_data.get("old_content")
            new_content = input_data.get("new_content")
            if old_content is None or new_content is None:
                return "Ошибка: Для режима 'replace' необходимы 'old_content' и 'new_content'."
            
            with open(path, "r", encoding="utf-8") as f:
                file_content = f.read()
            
            if old_content not in file_content:
                return f"Ошибка: Исходный фрагмент 'old_content' не найден в файле '{path}'."

            file_content = file_content.replace(old_content, new_content, 1)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(file_content)
            return f"Файл '{path}' успешно обновлен в режиме 'replace'."

        else: # overwrite or append
            content = input_data.get("content")
            if content is None:
                return f"Ошибка: Для режима '{mode}' обязателен аргумент 'content'."
            
            write_mode = "w" if mode == "overwrite" else "a"
            with open(path, write_mode, encoding="utf-8") as f:
                f.write(content)
            return f"Файл '{path}' успешно обновлен в режиме '{mode}'."

    except FileNotFoundError:
        return f"Ошибка: Файл не найден по пути '{path}'."
    except Exception as e:
        return f"Ошибка: Не удалось выполнить операцию с файлом '{path}': {e}"


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
        "description": "Рекурсивно выводит дерево файлов и директорий по указанному пути.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к директории для просмотра. По умолчанию '.'"}
            },
            "required": [],
        },
    },
}

edit_file_tool_def = {
    "type": "function",
    "function": {
        "name": "edit_file_tool",
        "description": "Создает, перезаписывает, добавляет или заменяет контент в файле. Режимы: 'overwrite', 'append', 'replace'.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Полный путь к файлу."},
                "mode": {"type": "string", "enum": ["overwrite", "append", "replace"], "description": "Режим записи."},
                "content": {"type": "string", "description": "Содержимое для 'overwrite' или 'append'."},
                "old_content": {"type": "string", "description": "Исходный фрагмент для 'replace'."},
                "new_content": {"type": "string", "description": "Новый фрагмент для 'replace'."},
            },
            "required": ["path", "mode"],
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

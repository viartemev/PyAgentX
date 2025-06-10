"""
Этот модуль определяет основного AI-агента, его логику и цикл взаимодействия с пользователем.
"""
from typing import Callable, Optional, List, Dict, Any
import os
import json
import logging
from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionMessage
from dotenv import load_dotenv
import inspect

# Новый, улучшенный системный промпт, превращающий агента в программиста.
SYSTEM_PROMPT = """
Ты — автономный AI-агент, способный выполнять сложные задачи, используя доступные инструменты.
Твоя цель — успешно завершить поставленную задачу, шаг за шагом.

# ПРАВИЛА ВЗАИМОДЕЙСТВИЯ:
1.  **АНАЛИЗ**: Внимательно изучи предоставленный тебе контекст и задачу.
2.  **ПЛАН**: Составь внутренний план действий. Какой инструмент использовать первым?
3.  **ДЕЙСТВИЕ**: Вызови выбранный инструмент с правильными аргументами.
4.  **РЕЗУЛЬТАТ**: Изучи результат вызова инструмента.
5.  **ЦИКЛ**: Повторяй шаги 1-4, пока не достигнешь конечной цели. Когда задача будет полностью выполнена, сообщи об этом, предоставив финальный результат.

# ВАЖНЫЕ ИНСТРУКЦИИ ПО РАБОТЕ:

## 1. Работа с файловой системой:
-   **Абсолютные импорты**: При написании Python кода всегда используй абсолютные импорты от корня проекта. Корень проекта - это директория, где лежит `main.py`. Например: `from app.agents.tools import read_file_tool`. **НИКОГДА** не используй относительные импорты (`from .. import ...`) или хаки с `sys.path`.
-   **Редактирование файлов**: Инструмент `edit_file_tool` имеет два режима:
    -   `mode='overwrite'` (по умолчанию): **Полностью перезаписывает** файл. Используй с осторожностью.
    -   `mode='append'`: **Добавляет контент в конец файла**. Используй этот режим, когда тебе нужно добавить новую функцию, класс или текст в уже существующий файл, не удаляя его содержимое.

## 2. Мышление и логика:
-   **Код прежде тестов**: Если твоя цель — написать функцию и тесты к ней, всегда сначала полностью реализуй **корректную и финальную** логику самой функции. Только после этого приступай к написанию тестов. Не тестируй заготовки или неполный код.
-   Будь методичен. Не торопись.
-   Если результат не соответствует ожиданиям, попробуй другой подход.
-   Если ты застрял, сделай шаг назад и пересмотри свой план.
"""

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Agent:
    """
    ИИ-агент, который может использовать инструменты (функции) для ответов на вопросы.
    """
    def __init__(
        self,
        name: str,
        api_key: str,
        model: str = "gpt-4-turbo",
    ):
        self.name = name
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.tools: Dict[str, Callable] = {}

    def add_tool(self, tool_func: Callable):
        """Добавляет инструмент (функцию) к агенту."""
        if not hasattr(tool_func, '__name__') or not tool_func.__name__:
            raise ValueError("Инструмент должен иметь имя (__name__).")
        
        self.tools[tool_func.__name__] = tool_func
    
    def get_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Возвращает список инструментов в формате OpenAI."""
        if not self.tools:
            return None
        
        openai_tools = []
        for name, func in self.tools.items():
            # Используем docstring как основное описание
            description = inspect.getdoc(func)

            # ВРЕМЕННОЕ РЕШЕНИЕ для определения параметров.
            # В будущем это можно улучшить, используя более строгий парсинг docstring'ов
            # или Pydantic модели.
            params = {}
            if name == "read_file_tool":
                params = {"path": {"type": "string", "description": "Полный путь к файлу, который нужно прочитать."}}
            elif name == "list_files_tool":
                params = {"path": {"type": "string", "description": "Путь к директории для просмотра. По умолчанию '.' (текущая директория)."}}
            elif name == "edit_file_tool":
                params = {
                    "path": {"type": "string", "description": "Полный путь к файлу для записи."},
                    "content": {"type": "string", "description": "Полное новое содержимое файла."}
                }
            elif name == "delete_file_tool":
                params = {"path": {"type": "string", "description": "Полный путь к файлу для удаления."}}
            
            # Определяем обязательные параметры
            required_params = []
            if name in ["read_file_tool", "edit_file_tool", "delete_file_tool"]:
                required_params = list(params.keys())
            if name == "edit_file_tool" and "content" not in required_params:
                 required_params.append("content")

            clean_name = name.replace('_tool', '')

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": clean_name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": params,
                        "required": required_params,
                    },
                },
            })
        return openai_tools
    
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Выполняет инструмент и возвращает его результат в виде строки."""
        full_tool_name = f"{tool_name}_tool"
        logging.info("Выполнение инструмента '%s' с аргументами: %s", full_tool_name, tool_args)
        tool_func = self.tools.get(full_tool_name)

        if not tool_func:
            logging.warning("Попытка вызова неизвестного инструмента: '%s'", full_tool_name)
            return f"Ошибка: Инструмент '{tool_name}' не найден."
        
        try:
            # Наши функции-инструменты ожидают один словарь в качестве аргумента
            result = tool_func(tool_args)
            logging.info("Инструмент '%s' успешно выполнен.", full_tool_name)
            return str(result)
        except Exception as e:
            logging.error("Ошибка при выполнении инструмента '%s': %s", full_tool_name, e, exc_info=True)
            return f"Ошибка: Не удалось выполнить инструмент '{tool_name}'. Причина: {e}"

    def _get_user_input(self) -> Optional[str]:
        """Обрабатывает получение ввода от пользователя."""
        print("\033[94mВы:\033[0m ", end="")
        try:
            user_input = input()
            return user_input if user_input.strip() else None
        except EOFError:
            return None

    def execute_task(self, briefing: str, task: str) -> str:
        """
        Выполняет конкретную задачу в неинтерактивном режиме.

        Args:
            briefing (str): Системный промпт, содержащий общий контекст и цель.
            task (str): Конкретная задача для выполнения.

        Returns:
            Строка с финальным результатом выполнения задачи.
        """
        logging.info("Агент '%s' получил новую задачу: %s", self.name, task)
        
        # Используем брифинг как системное сообщение, а задачу - как сообщение от пользователя
        self.messages = [
            {"role": "system", "content": briefing},
            {"role": "user", "content": f"Вот твоя задача: {task}. Выполни ее."}
        ]

        max_tool_calls = 5
        for i in range(max_tool_calls):
            logging.info("Шаг #%d. Вызов OpenAI с историей из %d сообщений.", i + 1, len(self.messages))
            
            try:
                response: ChatCompletionMessage = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.get_openai_tools(),
                    tool_choice="auto" if self.tools else None,
                ).choices[0].message
            except APIError as e:
                error_message = f"Ошибка OpenAI API при выполнении задачи: {e}"
                logging.error(error_message)
                return error_message

            if not response.tool_calls:
                final_answer = response.content or "Задача выполнена, но ответа от LLM не последовало."
                logging.info("--- Завершение подзадачи. Результат: %s ---", final_answer)
                return final_answer
            
            self.messages.append(response.model_dump())

            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                try:
                    tool_args = json.loads(tool_args_str)
                    tool_result = self._execute_tool(tool_name, tool_args)
                except json.JSONDecodeError:
                    logging.error("Не удалось декодировать аргументы для '%s': %s", tool_name, tool_args_str)
                    tool_result = f"Ошибка: Неверные аргументы для инструмента '{tool_name}'."
                
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_name,
                    "content": tool_result,
                })
        
        warning_message = "Достигнут лимит вызовов инструментов. Не удалось завершить задачу."
        logging.warning(warning_message)
        return warning_message

    def run(self) -> None:
        """Запускает основной цикл общения с агентом в интерактивном режиме."""
        print("Общение с агентом (используйте Ctrl+D или Ctrl+C для выхода)")
        conversation: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        try:
            while True:
                user_input = self._get_user_input()
                if user_input is None:
                    break
                
                # Здесь мы используем ту же логику, что и в execute_task
                # для выполнения пользовательского запроса.
                # Это можно будет в будущем вынести в отдельный метод.
                conversation.append({"role": "user", "content": user_input})

                max_tool_calls = 5
                for _ in range(max_tool_calls):
                    response: ChatCompletionMessage = self.client.chat.completions.create(
                        model=self.model,
                        messages=conversation,
                        tools=self.get_openai_tools(),
                    ).choices[0].message
                    
                    if not response.tool_calls:
                        assistant_response = response.content or ""
                        print(f"\033[93m{self.name}:\033[0m {assistant_response}")
                        conversation.append({"role": "assistant", "content": assistant_response})
                        break
                    
                    conversation.append(response.model_dump())

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        tool_result = self._execute_tool(tool_name, tool_args)
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": tool_result,
                        })
                else:
                    print(f"\033[91m{self.name}: Достигнут лимит вызовов инструментов.\033[0m")
        
        except (KeyboardInterrupt, EOFError):
            print("\nВыход из чата.")


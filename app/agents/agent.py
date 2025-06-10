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
        model: str = "o4-mini",
        max_iterations: int = 10,
    ):
        self.name = name
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_iterations = max_iterations
        self.system_prompt = "Ты — универсальный AI-ассистент."

    def add_tool(self, tool_func: Callable, tool_definition: Dict[str, Any]):
        """Добавляет инструмент и его определение."""
        tool_name = tool_func.__name__
        self.tools[tool_name] = tool_func
        self.tool_definitions.append(tool_definition)
    
    def get_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Возвращает список определений инструментов для OpenAI API."""
        if not self.tool_definitions:
            return None
        return self.tool_definitions
    
    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Выполняет указанный инструмент с аргументами."""
        if tool_name in self.tools:
            try:
                # Наши функции-инструменты ожидают один словарь в качестве аргумента
                result = self.tools[tool_name](tool_args)
                return result
            except Exception as e:
                logging.error("Ошибка при выполнении инструмента '%s': %s", tool_name, e, exc_info=True)
                return f"Ошибка: Не удалось выполнить инструмент '{tool_name}'. Причина: {e}"
        else:
            logging.warning("Попытка вызова неизвестного инструмента: '%s'", tool_name)
            return f"Ошибка: Инструмент '{tool_name}' не найден."

    def _get_user_input(self) -> Optional[str]:
        """Обрабатывает получение ввода от пользователя."""
        print("\033[94mВы:\033[0m ", end="")
        try:
            user_input = input()
            return user_input if user_input.strip() else None
        except EOFError:
            return None

    def _get_model_response(self) -> ChatCompletionMessage:
        """
        Отправляет текущую историю беседы в OpenAI и возвращает ответ модели.
        """
        try:
            logging.info(f"Отправка запроса в OpenAI с {len(self.conversation_history)} сообщениями.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                tools=self.get_openai_tools(),
                tool_choice="auto",
            )
            return response.choices[0].message
        except Exception as e:
            logging.error(f"Ошибка при вызове OpenAI API: {e}", exc_info=True)
            # Возвращаем "пустое" сообщение с контентом об ошибке, чтобы цикл мог его обработать
            return ChatCompletionMessage(role="assistant", content=f"Произошла ошибка API: {e}")

    def execute_task(self, briefing: str) -> str:
        """
        Выполняет одну задачу на основе предоставленного брифинга.
        """
        logging.info(f"Агент {self.name} получил задачу.")
        
        # Системный промпт определяет "личность" агента, а брифинг - контекст задачи.
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": briefing}
        ]
        
        for _ in range(self.max_iterations):
            response_message = self._get_model_response()

            if response_message.content and "ошибка API" in response_message.content.lower():
                return response_message.content

            if not response_message.tool_calls:
                final_answer = response_message.content or "Задача выполнена."
                logging.info(f"Агент {self.name} завершил задачу с ответом: {final_answer}")
                return final_answer
            
            self.conversation_history.append(response_message)
            
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args_str = tool_call.function.arguments
                logging.info(f"Вызов инструмента: {tool_name} с аргументами: {tool_args_str}")
                
                try:
                    tool_args = json.loads(tool_args_str)
                    tool_result = self._execute_tool(tool_name, tool_args)
                except json.JSONDecodeError:
                    error_msg = f"Ошибка: неверный JSON в аргументах инструмента: {tool_args_str}"
                    logging.error(error_msg)
                    tool_result = error_msg
                
                self.conversation_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": str(tool_result),
                    }
                )
        
        warning_message = f"Агент {self.name} достиг лимита итераций ({self.max_iterations}), не завершив задачу."
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


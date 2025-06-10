from typing import Callable, Tuple, Optional, List, Dict, Any
import os
import json
import logging
from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from dotenv import load_dotenv
from app.agents.tools import ToolDefinition

# Четкий и инструктивный системный промпт
SYSTEM_PROMPT = (
    "Ты — полезный ИИ-ассистент. У тебя есть доступ к набору инструментов для ответов на вопросы пользователя. "
    "Когда пользователь задает вопрос, сначала подумай, нужно ли использовать инструмент. "
    "Если решишь использовать инструмент, вызови его. После получения результата от инструмента, "
    "используй этот результат для формулирования окончательного ответа пользователю. "
    "Не просто констатируй вывод инструмента, а дай полезный и развернутый ответ, который включает в себя полученные данные."
)

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
        user_input_handler: Optional[Callable[[], Tuple[str, bool]]] = None,
        tools: Optional[List[ToolDefinition]] = None
    ):
        load_dotenv()
        self.name = name
        self.user_input_handler = user_input_handler
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o") # Используем более способную модель по умолчанию
        self.client = OpenAI(api_key=self.api_key)

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Выполняет инструмент и возвращает его результат в виде строки."""
        logging.info(f"Выполнение инструмента '{tool_name}' с аргументами: {tool_args}")
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            try:
                result = tool.function(tool_args)
                logging.info(f"Инструмент '{tool_name}' успешно выполнен.")
                return str(result)
            except Exception as e:
                logging.error(f"Ошибка при выполнении инструмента '{tool_name}': {e}", exc_info=True)
                return f"Ошибка: Не удалось выполнить инструмент '{tool_name}'. Причина: {e}"
        else:
            logging.warning(f"Попытка вызова неизвестного инструмента: '{tool_name}'")
            return f"Ошибка: Инструмент '{tool_name}' не найден."

    def _get_user_input(self) -> Optional[str]:
        """Обрабатывает получение ввода от пользователя."""
        print("\033[94mВы:\033[0m ", end="")
        if self.user_input_handler:
            user_input, ok = self.user_input_handler()
            return user_input if ok else None
        else:
            try:
                user_input = input()
                return user_input if user_input.strip() else None
            except EOFError:
                return None

    def run(self) -> None:
        """Запускает основной цикл общения с агентом."""
        print("Общение с агентом (используйте Ctrl+D или Ctrl+C для выхода)")
        conversation: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

        try:
            while True:
                user_input = self._get_user_input()
                if user_input is None:
                    break

                conversation.append({"role": "user", "content": user_input})

                max_tool_calls = 5
                for _ in range(max_tool_calls):
                    logging.info(f"Вызов OpenAI с историей из {len(conversation)} сообщений:\n{json.dumps(conversation, indent=2, ensure_ascii=False)}")
                    
                    response: ChatCompletionMessage = self.client.chat.completions.create(
                        model=self.model,
                        messages=conversation,
                        tools=[
                            {"type": "function", "function": tool.to_openai_spec()}
                            for tool in self.tools.values()
                        ] if self.tools else None,
                        tool_choice="auto" if self.tools else None,
                    ).choices[0].message
                    
                    if not response.tool_calls:
                        assistant_response = response.content or ""
                        print(f"\033[93m{self.name}:\033[0m {assistant_response}")
                        conversation.append({"role": "assistant", "content": assistant_response})
                        break
                    
                    conversation.append(response.model_dump())

                    for tool_call in response.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args_str = tool_call.function.arguments
                        try:
                            tool_args = json.loads(tool_args_str)
                            tool_result = self._execute_tool(tool_name, tool_args)
                        except json.JSONDecodeError:
                            logging.error(f"Не удалось декодировать аргументы для '{tool_name}': {tool_args_str}")
                            tool_result = f"Ошибка: Неверные аргументы для инструмента '{tool_name}'."
                        
                        conversation.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": tool_result,
                        })
                else:
                    logging.warning("Достигнут лимит вызовов инструментов. Цикл прерван.")
                    print(f"\033[91m{self.name}: Кажется, я застрял в цикле использования инструментов. Давай попробуем что-нибудь другое.\033[0m")
        
        except (KeyboardInterrupt, EOFError):
            print("\nВыход из чата.")


from typing import Callable, Tuple, Optional, List, Dict, Any
import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
from app.agents.tools import ToolDefinition

SYSTEM_PROMPT = ("You are an AI assistant")

class Agent:
    """AI-агент с поддержкой инструментов (OpenAI function calling).

    Args:
        name (str): Имя агента.
        user_input (Optional[Callable[[], Tuple[str, bool]]]): Функция для пользовательского ввода.
        tools (Optional[List[ToolDefinition]]): Список инструментов.
    """
    def __init__(self, name: str, user_input: Optional[Callable[[], Tuple[str, bool]]] = None, tools: Optional[List[ToolDefinition]] = None):
        load_dotenv()
        self.name = name
        self.user_input = user_input
        self.model = os.getenv("OPENAI_MODEL", "o4-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.tools = tools or []
        logging.basicConfig(level=logging.INFO)

    def process(self, input: str) -> str:
        """Основная логика обработки входных данных агентом."""
        return f"Agent {self.name} processed: {input}"

    def execute_tool(self, name: str, input_data: Dict[str, Any]) -> str:
        """Выполняет инструмент по имени с заданными аргументами."""
        for tool in self.tools:
            if tool.name == name:
                try:
                    result = tool.function(input_data)
                    logging.info(f"Tool '{name}' executed with input {input_data}. Result: {result}")
                    return result
                except Exception as e:
                    logging.error(f"Tool '{name}' failed: {e}")
                    return f"[Tool '{name}' failed: {e}]"
        logging.error(f"Tool '{name}' not found")
        return f"[Tool '{name}' not found]"

    def run_inference(self, conversation: List[Dict[str, Any]]) -> Any:
        """Выполняет запрос к OpenAI Chat API, используя историю сообщений и инструменты."""
        messages = []
        for msg in conversation:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "agent":
                messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "system":
                messages.append({"role": "system", "content": msg["content"]})
            elif msg["role"] == "assistant":
                messages.append({"role": "assistant", "content": msg["content"]})
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.tools
        ] if self.tools else None
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                max_completion_tokens=1024,
            )
            message = response.choices[0].message
            # Если модель вызывает функцию
            if hasattr(message, "tool_calls") and message.tool_calls:
                results = []
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_result = self.execute_tool(tool_name, tool_args)
                    results.append({
                        "role": "function",
                        "name": tool_name,
                        "content": tool_result
                    })
                return results[0] if results else {"role": "agent", "content": "(no tool result)"}
            else:
                return {"role": "agent", "content": message.content}
        except Exception as e:
            logging.error(f"OpenAI error: {e}")
            return {"role": "agent", "content": f"[OpenAI error: {e}]"}

    def run(self) -> None:
        """Интерактивный чат-цикл с поддержкой инструментов, system prompt и защитой от бесконечных вызовов."""
        conversation: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        print("Chat with Agent (use Ctrl-C to quit)")
        try:
            while True:
                print("\u001b[94mYou\u001b[0m: ", end="")
                if self.user_input:
                    user_input, ok = self.user_input()
                    if not ok:
                        break
                else:
                    user_input = input()
                    ok = True
                if not ok or not user_input.strip():
                    break

                user_message = {"role": "user", "content": user_input}
                conversation.append(user_message)

                while True:
                    agent_message = self.run_inference(conversation)
                    print(f"Agent message: {agent_message}")

                    if agent_message.get("role") == "agent":
                        conversation.append(agent_message)
                        print(f"\u001b[93m{self.name}\u001b[0m: {agent_message.get('content')}")
                        break
                    elif agent_message.get("role") == "function":
                        print(f"\u001b[92mtool\u001b[0m: {agent_message.get('name')} => {agent_message.get('content')}")
                        # Формируем естественный результат для LLM
                        tool_result_message = {
                            "role": "function",
                            "name": agent_message["name"],
                            "content": agent_message["content"]
                        }
                        conversation.append(tool_result_message)
                        agent_message = self.run_inference(conversation)
                        conversation.append(agent_message)
                        print(f"\u001b[93m{self.name}\u001b[0m: {agent_message.get('content')}")
                        break
                    else:
                        conversation.append(agent_message)
                        print(f"\u001b[93m{self.name}\u001b[0m: (no response)")
                        break
        except KeyboardInterrupt:
            print("\nExiting chat.")


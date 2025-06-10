from typing import Callable, Tuple, Optional, List, Dict, Any
import os
from openai import OpenAI
from dotenv import load_dotenv

class Agent:
    """Базовый класс AI-агента.

    Args:
        name (str): Имя агента.
        user_input (Optional[Callable[[], Tuple[str, bool]]]): Функция для получения пользовательского ввода.

    Example:
        >>> def mock_input():
        ...     return ("Hello!", True)
        >>> agent = Agent(name="TestAgent", user_input=mock_input)
        >>> agent.run()
    """
    def __init__(self, name: str, user_input: Optional[Callable[[], Tuple[str, bool]]] = None):
        load_dotenv()
        self.name = name
        self.user_input = user_input
        self.model = os.getenv("OPENAI_MODEL", "o4-mini")
        self.api_key = os.getenv("OPENAI_API_KEY")

    def process(self, input: str) -> str:
        """Основная логика обработки входных данных агентом."""
        return f"Agent {self.name} processed: {input}"

    def run_inference(self, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Выполняет запрос к OpenAI Chat API, используя историю сообщений.

        Args:
            conversation (List[Dict[str, Any]]): История сообщений (user/agent).
        Returns:
            Dict[str, Any]: Сообщение агента в формате {"role": "agent", "content": str}
        """
        messages = []
        for msg in conversation:
            if msg["role"] == "user":
                messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "agent":
                messages.append({"role": "assistant", "content": msg["content"]})
        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=1024
            )
            content = response.choices[0].message.content
            return {"role": "agent", "content": content}
        except Exception as e:
            return {"role": "agent", "content": f"[OpenAI error: {e}]"}

    def run(self) -> None:
        """Интерактивный чат-цикл с историей сообщений, аналог Go-примера."""
        conversation: List[Dict[str, str]] = []
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
                agent_message = self.run_inference(conversation)
                conversation.append(agent_message)
                if agent_message.get("role") == "agent":
                    print(f"\u001b[93m{self.name}\u001b[0m: {agent_message.get('content')}")
                else:
                    print(f"\u001b[93m{self.name}\u001b[0m: (no response)")
        except KeyboardInterrupt:
            print("\nExiting chat.")


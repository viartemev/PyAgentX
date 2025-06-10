from typing import Callable, Tuple, Optional

class Agent:
    """Базовый класс AI-агента.

    Args:
        name (str): Имя агента.
        user_input (Optional[Callable[[], Tuple[str, bool]]]): Функция для получения пользовательского ввода.

    Example:
        >>> def mock_input():
        ...     return ("Hello!", True)
        >>> agent = Agent(name="TestAgent", user_input=mock_input)
        >>> agent.run("Привет!")
    """
    def __init__(self, name: str, user_input: Optional[Callable[[], Tuple[str, bool]]] = None):
        self.name = name
        self.user_input = user_input

    def process(self, input: str) -> str:
        """Основная логика обработки входных данных агентом."""
        return f"Agent {self.name} processed: {input}"

    def run(self, input: str) -> str:
        """Запуск агента с заданным вводом."""
        print(f"Agent {self.name} is running with input: {input}")
        return self.process(input)


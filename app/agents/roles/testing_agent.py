"""Testing Agent."""
from app.agents.agent import Agent

class TestingAgent(Agent):
    """Агент, специализирующийся на запуске тестов."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """
Ты — TestingAgent, автоматизированный робот для запуска тестов.
Твоя единственная задача — вызывать инструмент `run_tests_tool` с правильным путем к тестам.
После получения результата ты должен кратко и точно доложить об успехе или провале.
""" 
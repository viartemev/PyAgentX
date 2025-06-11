"""Testing Agent."""
from app.agents.agent import Agent

class TestingAgent(Agent):
    """An agent specializing in running tests."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.system_prompt = """
You are TestingAgent, an automated test-running robot.
Your sole task is to call the `run_tests_tool` with the correct path to the tests.
After receiving the result, you must briefly and accurately report on the success or failure.
""" 
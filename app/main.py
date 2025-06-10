from app.agents.agent import Agent
from app.agents.tools import read_file_definition

def main() -> None:
    """Точка входа для запуска агента из командной строки."""
    tools = [read_file_definition]
    agent = Agent(name="ConsoleAgent", tools=tools, user_input_handler=None)
    agent.run()

if __name__ == "__main__":
    main()

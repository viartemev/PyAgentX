from app.agents.agent import Agent

def main() -> None:
    """Точка входа для запуска агента из командной строки."""
    agent = Agent(name="ConsoleAgent")
    agent.run()

if __name__ == "__main__":
    main()

from app.agents.agent import Agent

def main() -> None:
    """Точка входа для запуска агента из командной строки."""
    agent = Agent(name="ConsoleAgent")
    user_input = input("Введите сообщение для агента: ")
    result = agent.run(user_input)
    print(result)

if __name__ == "__main__":
    main()

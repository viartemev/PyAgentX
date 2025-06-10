import logging
import os
from dotenv import load_dotenv
from app.agents.agent import Agent
from app.agents.tools import (
    list_files_tool,
    read_file_tool,
    edit_file_tool,
    delete_file_tool,
    run_tests_tool
)
from app.orchestration.decomposer import TaskDecomposer
from app.orchestration.orchestrator import Orchestrator

def main():
    """Главная функция для запуска AI агента."""
    load_dotenv()

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        handlers=[
            logging.FileHandler("agent_activity.log"),
            logging.StreamHandler()
        ]
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("Не найден ключ OPENAI_API_KEY в .env файле.")
        print("Ошибка: Пожалуйста, убедитесь, что ваш .env файл содержит OPENAI_API_KEY.")
        return

    try:
        # 1. Инициализация команды агентов
        logging.info("Инициализация команды агентов...")
        
        # Агент-кодер
        coding_agent = Agent(name="CodingAgent", api_key=api_key, model="o4-mini")
        coding_agent.add_tool(list_files_tool)
        coding_agent.add_tool(read_file_tool)
        coding_agent.add_tool(edit_file_tool)

        # Агент-тестировщик
        testing_agent = Agent(name="TestingAgent", api_key=api_key, model="o4-mini")
        testing_agent.add_tool(read_file_tool) # Чтобы читать код тестов
        testing_agent.add_tool(run_tests_tool) # Чтобы запускать тесты

        # Агент-оценщик (пока с базовым набором инструментов)
        evaluator_agent = Agent(name="EvaluatorAgent", api_key=api_key, model="o4-mini")
        evaluator_agent.add_tool(read_file_tool)

        # Агент-ревьюер, который проверяет качество кода
        reviewer_agent = Agent(name="ReviewerAgent", api_key=api_key, model="gpt-4-turbo")
        reviewer_agent.add_tool(read_file_tool)

        # Создаем словарь рабочих агентов для Оркестратора
        workers = {
            "CodingAgent": coding_agent,
            "TestingAgent": testing_agent,
            "EvaluatorAgent": evaluator_agent,
            "ReviewerAgent": reviewer_agent,
        }
        # Добавляем универсального агента, если задача не назначена конкретному
        workers["DefaultAgent"] = Agent(name="DefaultAgent", api_key=api_key, model="o4-mini")
        workers["DefaultAgent"].add_tool(list_files_tool)
        workers["DefaultAgent"].add_tool(read_file_tool)

        # Планировщик
        planner = TaskDecomposer(api_key=api_key, model="o4-mini")
        
        # Оркестратор теперь управляет командой
        orchestrator = Orchestrator(
            task_decomposer=planner,
            worker_agents=workers
        )

        # 2. Запрос цели и запуск оркестратора
        goal = input("Пожалуйста, введите вашу высокоуровневую цель: ")
        if not goal:
            print("Цель не может быть пустой.")
            return

        orchestrator.run(goal)

    except Exception as e:
        logging.critical("Произошла критическая ошибка: %s", e, exc_info=True)
        print(f"\nКритическая ошибка: {e}")

if __name__ == "__main__":
    main() 
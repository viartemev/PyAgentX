import logging
import os
from dotenv import load_dotenv
from app.agents.agent import Agent
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    edit_file_tool, edit_file_tool_def,
    list_files_tool, list_files_tool_def,
    run_tests_tool, run_tests_tool_def,
    subtract_tool, subtract_tool_def
)
from app.orchestration.decomposer import TaskDecomposer
from app.orchestration.orchestrator import Orchestrator
from app.agents.specialized_agents import (
    CodingAgent,
    ReviewerAgent,
    TestingAgent,
    EvaluatorAgent,
)

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
        
        common_kwargs = {"api_key": api_key, "model": "o4-mini"}

        coding_agent = CodingAgent(name="CodingAgent", **common_kwargs)
        coding_agent.add_tool(read_file_tool, read_file_tool_def)
        coding_agent.add_tool(edit_file_tool, edit_file_tool_def)
        coding_agent.add_tool(list_files_tool, list_files_tool_def)
        coding_agent.add_tool(subtract_tool, subtract_tool_def)

        testing_agent = TestingAgent(name="TestingAgent", **common_kwargs)
        testing_agent.add_tool(read_file_tool, read_file_tool_def)
        testing_agent.add_tool(run_tests_tool, run_tests_tool_def)

        evaluator_agent = EvaluatorAgent(name="EvaluatorAgent", **common_kwargs)
        evaluator_agent.add_tool(read_file_tool, read_file_tool_def)

        reviewer_agent = ReviewerAgent(name="ReviewerAgent", **common_kwargs)
        reviewer_agent.add_tool(read_file_tool, read_file_tool_def)

        workers = {
            "CodingAgent": coding_agent,
            "TestingAgent": testing_agent,
            "EvaluatorAgent": evaluator_agent,
            "ReviewerAgent": reviewer_agent,
        }
        
        default_agent = Agent(name="DefaultAgent", **common_kwargs)
        default_agent.add_tool(list_files_tool, list_files_tool_def)
        default_agent.add_tool(read_file_tool, read_file_tool_def)
        workers["DefaultAgent"] = default_agent

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
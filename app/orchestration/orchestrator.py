"""
Модуль Оркестратора, который управляет выполнением плана задач.
"""
import logging
import json
from typing import Dict, Any, List, Tuple, Optional
from app.agents.agent import Agent
from app.orchestration.decomposer import TaskDecomposer

class Orchestrator:
    """
    Оркестратор управляет всем процессом: от декомпозиции цели до выполнения задач агентами.
    """
    def __init__(self, task_decomposer: TaskDecomposer, worker_agents: Dict[str, Agent]):
        self.task_decomposer = task_decomposer
        self.worker_agents = worker_agents
        self.previous_steps_results: List[str] = []

    def _get_briefing(self, current_task: Dict[str, Any], plan: List[Dict[str, Any]], goal: str) -> str:
        """Создает полный контекст (брифинг) для агента перед выполнением задачи."""
        briefing = f"КОНТЕКСТ ВЫПОЛНЕНИЯ ЗАДАЧИ\n"
        briefing += "-" * 30 + "\n"
        briefing += f"**Глобальная цель:** {goal}\n\n"
        briefing += f"**Весь план:**\n"
        for step in plan:
            marker = "-->" if step == current_task else "   "
            briefing += f"{marker} Шаг {step['step']}: {step['task']} (Исполнитель: {step['assignee']})\n"
        
        briefing += f"\n**Текущая задача для тебя:**\n- {current_task['task']}\n- {current_task['description']}\n\n"

        if self.previous_steps_results:
            briefing += "**Результаты предыдущих шагов:**\n"
            briefing += "\n".join(self.previous_steps_results)
        else:
            briefing += "**Это первый шаг, предыдущих результатов нет.**\n"
        
        briefing += "\n" + "-" * 30 + "\n"
        briefing += "Пожалуйста, выполни свою задачу, используя доступные тебе инструменты. Будь краток и точен в своих действиях."
        
        return briefing

    def _perform_code_review(self, task: Dict[str, Any], plan: List[Dict[str, Any]], goal: str) -> Tuple[bool, Optional[str]]:
        """Выполняет шаг Code Review."""
        logging.info("--- Начат этап Code Review ---")
        reviewer_agent = self.worker_agents.get("ReviewerAgent")
        if not reviewer_agent:
            logging.error("Агент ReviewerAgent не найден!")
            return False, "Агент ReviewerAgent не был инициализирован."

        briefing = self._get_briefing(task, plan, goal)
        review_result = reviewer_agent.execute_task(briefing)

        logging.info(f"Результат Code Review: {review_result}")

        if "LGTM" in review_result.upper():
            logging.info("--- Code Review пройдено успешно ---")
            return True, None
        else:
            logging.warning("--- Code Review выявило проблемы ---")
            return False, review_result

    def run(self, goal: str):
        self.previous_steps_results = []
        plan = self._get_plan(goal)
        current_step_index = 0

        while current_step_index < len(plan):
            task = plan[current_step_index]
            logging.info(f"--- Выполнение шага {task['step']}: {task['task']} ---")
            
            if task.get("assignee") == "ReviewerAgent":
                review_passed, suggestions = self._perform_code_review(task, plan, goal)
                
                if review_passed:
                    self.previous_steps_results.append(
                        f"Шаг {task['step']}: {task['task']} - Успешно (LGTM)"
                    )
                    current_step_index += 1
                    continue
                else:
                    logging.warning("Ревью провалено. Создание задачи на исправление...")
                    new_task_description = (
                        "Агент-ревьюер нашел проблемы в коде, который ты написал. "
                        f"Вот его комментарии: '{suggestions}'. "
                        "Твоя задача — исправить код в соответствии с этими рекомендациями."
                    )
                    
                    new_task = {
                        "step": task["step"] + 0.1,
                        "assignee": "CodingAgent",
                        "task": "Исправить код на основе замечаний Code Review.",
                        "description": new_task_description,
                    }
                    
                    plan.insert(current_step_index + 1, new_task)
                    logging.info(f"В план добавлена новая задача: {new_task}")
                    
                    self.previous_steps_results.append(
                        f"Шаг {task['step']}: {task['task']} - Провалено. Замечания: {suggestions}"
                    )
                    current_step_index += 1
                    continue

            assignee_name = task.get("assignee", "DefaultAgent")
            agent = self.worker_agents.get(assignee_name)
            
            if not agent:
                logging.error(f"Агент {assignee_name} не найден! Пропускаем шаг.")
                self.previous_steps_results.append(
                    f"Шаг {task['step']}: {task['task']} - Пропущен (агент не найден)"
                )
                current_step_index += 1
                continue

            briefing = self._get_briefing(task, plan, goal)
            result = agent.execute_task(briefing)

            logging.info(f"Результат шага {task['step']}: {result}")
            self.previous_steps_results.append(
                f"Шаг {task['step']}: {task['task']} - Результат: {result}"
            )

            if agent.name == "TestingAgent" and any(keyword in result.lower() for keyword in ["failed", "error", "traceback"]):
                logging.error("Тесты провалены! Запускаем процесс исправления.")
                # TODO: Implement test failure handling
                pass

            current_step_index += 1

        logging.info("Все задачи выполнены. Цель достигнута.")

    def _get_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Получает план от TaskDecomposer."""
        logging.info("Получение плана от TaskDecomposer...")
        plan = self.task_decomposer.generate_plan(goal)
        if not plan:
            logging.error("Не удалось составить план. Оркестратор останавливает работу.")
            return []
        
        logging.info("План успешно получен.")
        print("\nСоставлен следующий план:")
        for step in plan:
            print(f"- Шаг {step['step']}: {step['task']} (Исполнитель: {step['assignee']})")
        return plan

    # def _create_evaluator_briefing(self, failed_test_result: str, last_briefing: str) -> str:
    #     """Создает системный промпт для EvaluatorAgent."""
    #     return f"""
    # ... (здесь был промпт)
    # """ 
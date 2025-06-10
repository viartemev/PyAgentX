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
        self.execution_history: List[Dict[str, Any]] = []

    def _get_briefing(self, current_task: Dict[str, Any], plan: List[Dict[str, Any]], goal: str) -> str:
        """Создает полный контекст (брифинг) для агента перед выполнением задачи."""
        
        # Формирование истории предыдущих шагов
        history_log = ""
        if not self.execution_history:
            history_log = "Это первый шаг, предыдущих результатов нет.\n"
        else:
            history_log += "**История выполнения:**\n"
            for record in self.execution_history:
                history_log += f"- **Шаг {record['step']} ({record['assignee']})**: {record['task']}\n"
                history_log += f"  - **Результат**: {record['result']}\n"

        # Формирование полного плана
        plan_log = ""
        for step in plan:
            marker = "-->" if step['step'] == current_task['step'] else "   "
            plan_log += f"{marker} Шаг {step['step']}: {step['task']} (Исполнитель: {step['assignee']})\n"

        # Финальный брифинг
        briefing = f"""
КОНТЕКСТ ЗАДАЧИ
---------------------------------
**Глобальная цель:** {goal}

**ВЕСЬ ПЛАН ЗАДАЧ:**
{plan_log}
**ИСТОРИЯ ВЫПОЛНЕНИЯ:**
{history_log}
---------------------------------
**ВАША ТЕКУЩАЯ ЗАДАЧА (Шаг {current_task['step']}):**

**Задача:** {current_task['task']}
**Описание:** {current_task['description']}

Проанализируйте всю предоставленную информацию, особенно результаты предыдущих шагов, и выполните свою задачу.
"""
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
        self.execution_history = []
        plan = self._get_plan(goal)
        current_step_index = 0

        while current_step_index < len(plan):
            task = plan[current_step_index]
            logging.info(f"--- Выполнение шага {task['step']}: {task['task']} ---")

            if task.get("assignee") == "ReviewerAgent":
                review_passed, suggestions = self._perform_code_review(task, plan, goal)
                
                history_record = {
                    "step": task['step'],
                    "task": task['task'],
                    "assignee": task['assignee'],
                }

                if review_passed:
                    history_record["result"] = "Успешно (LGTM)."
                    self.execution_history.append(history_record)
                    current_step_index += 1
                    continue
                else:
                    history_record["result"] = f"Провалено. Замечания: {suggestions}"
                    self.execution_history.append(history_record)
                    
                    logging.warning("Ревью провалено. Создание задачи на исправление...")
                    new_task_description = (
                        "Агент-ревьюер нашел проблемы в коде, который ты написал. "
                        f"Вот его комментарии: '{suggestions}'. "
                        "Твоя задача — исправить код в соответствии с этими рекомендациями."
                    )
                    
                    new_task = {
                        "step": float(task["step"]) + 0.1,
                        "assignee": "CodingAgent",
                        "task": "Исправить код на основе замечаний Code Review.",
                        "description": new_task_description,
                    }
                    
                    plan.insert(current_step_index + 1, new_task)
                    logging.info(f"В план добавлена новая задача: {new_task}")
                    
                    current_step_index += 1
                    continue

            assignee_name = task.get("assignee", "DefaultAgent")
            agent = self.worker_agents.get(assignee_name)
            
            if not agent:
                logging.error(f"Агент {assignee_name} не найден! Пропускаем шаг.")
                result = f"Пропущено (агент '{assignee_name}' не найден)."
            else:
                briefing = self._get_briefing(task, plan, goal)
                result = agent.execute_task(briefing)

            logging.info(f"Результат шага {task['step']}: {result}")
            
            history_record = {
                "step": task['step'],
                "task": task['task'],
                "assignee": task.get("assignee", "DefaultAgent"),
                "result": result
            }
            self.execution_history.append(history_record)

            if agent and agent.name == "TestingAgent" and "ПРОВАЛ" in result.upper():
                logging.error("Тесты провалены! Запускаем процесс исправления.")
                evaluator_agent = self.worker_agents.get("EvaluatorAgent")
                if not evaluator_agent:
                    logging.error("EvaluatorAgent не найден, невозможно проанализировать ошибку.")
                    current_step_index += 1
                    continue

                evaluator_briefing = (
                    "Тесты провалились. Проанализируй следующий лог и сформулируй задачу по его исправлению.\n\n"
                    "ЛОГ ОШИБКИ:\n"
                    f"{result}"
                )
                
                fix_task_description = evaluator_agent.execute_task(evaluator_briefing)
                logging.info(f"EvaluatorAgent предложил следующую задачу: {fix_task_description}")

                new_task = {
                    "step": float(task["step"]) + 0.1,
                    "assignee": "CodingAgent",
                    "task": "Исправить код на основе проваленных тестов.",
                    "description": fix_task_description,
                }
                
                plan.insert(current_step_index + 1, new_task)
                logging.info(f"В план добавлена новая задача на исправление: {new_task}")

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
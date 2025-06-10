"""
Модуль Оркестратора, который управляет выполнением плана задач.
"""
import logging
import json
from typing import Dict, Any, List
from app.agents.agent import Agent
from app.orchestration.decomposer import TaskDecomposer

class Orchestrator:
    """
    Оркестратор управляет выполнением высокоуровневой цели,
    декомпозируя ее на задачи и назначая их соответствующим агентам-воркерам.
    """

    def __init__(self, task_decomposer: TaskDecomposer, worker_agents: Dict[str, Agent]):
        """
        Инициализирует Orchestrator.

        Args:
            task_decomposer (TaskDecomposer): Экземпляр планировщика задач.
            worker_agents (Dict[str, Agent]): Словарь с доступными агентами-воркерами,
                                              где ключ - имя агента, значение - объект Agent.
        """
        self.task_decomposer = task_decomposer
        self.worker_agents = worker_agents

    def run(self, goal: str):
        """
        Запускает процесс выполнения цели.

        Args:
            goal (str): Высокоуровневая цель, поставленная пользователем.
        """
        logging.info("Оркестратор получил цель: %s", goal)

        # 1. Декомпозиция цели на план
        plan = self.task_decomposer.generate_plan(goal)
        if not plan:
            logging.error("Не удалось составить план. Оркестратор останавливает работу.")
            return

        print("\nСоставлен следующий план:")
        for step in plan:
            print(f"- Шаг {step['step']}: {step['task']} (Исполнитель: {step['assignee']})")

        # 2. Выполнение плана
        results = []
        tasks_to_run = list(plan) # Создаем копию, которую можно изменять
        
        i = 0
        while i < len(tasks_to_run):
            task = tasks_to_run[i]
            print(f"\n{'='*20} Шаг {i+1}/{len(tasks_to_run)} {'='*20}")
            print(f"ЗАДАЧА: {task['task']}")
            print(f"ИСПОЛНИТЕЛЬ: {task['assignee']}")
            print(f"{'-'*55}")

            # Выбираем правильного агента
            assignee_name = task.get("assignee", "DefaultAgent")
            worker_agent = self.worker_agents.get(assignee_name)

            if not worker_agent:
                logging.error(f"Агент с именем '{assignee_name}' не найден! Пропускаем шаг.")
                results.append({"step": i + 1, "task": task['task'], "result": "Ошибка: Исполнитель не найден."})
                i += 1
                continue
            
            # Формируем "брифинг" для агента
            briefing = self._create_briefing(goal, tasks_to_run, task, results)
            
            # Выполняем задачу
            result = worker_agent.execute_task(briefing, task['task'])
            results.append({"step": i + 1, "task": task['task'], "result": result})
            
            logging.info("Результат шага %d: %s", i + 1, result)
            print(f"\nРЕЗУЛЬТАТ ШАГА:\n{result}")

            # --- НАЧАЛО НОВОЙ ЛОГИКИ: ЦИКЛ ОБРАТНОЙ СВЯЗИ ---
            is_test_step = task.get("assignee") == "TestingAgent"
            # Более надежная проверка на провал
            failure_keywords = ["ПРОВАЛ", "ОШИБКА", "error", "failed", "не увенчалась успехом"]
            test_failed = is_test_step and any(keyword in result for keyword in failure_keywords)

            if test_failed:
                print("\n\033[1;33m>>> ОБНАРУЖЕНА ОШИБКА В ТЕСТАХ. ЗАПУСК ЦИКЛА ОТЛАДКИ.\033[0m")
                
                evaluator_agent = self.worker_agents.get("EvaluatorAgent")
                if not evaluator_agent:
                    logging.error("EvaluatorAgent не найден! Невозможно запустить цикл отладки.")
                    i += 1
                    continue
                
                # Создаем специальный промпт для "мозга" EvaluatorAgent
                evaluator_briefing = self._create_evaluator_briefing(result, briefing)
                
                print("\n\033[1;35m>>> Вызов EvaluatorAgent для анализа ошибки...\033[0m")
                new_task_json_str = evaluator_agent.execute_task(
                    evaluator_briefing,
                    "Проанализируй провал тестов и верни ТОЛЬКО JSON с новой задачей для CodingAgent."
                )

                try:
                    # Парсим ответ от EvaluatorAgent
                    new_task_data = json.loads(new_task_json_str)
                    new_task = {
                        "step": len(tasks_to_run) + 1,
                        "assignee": "CodingAgent", # Задача всегда для кодера
                        "task": new_task_data.get("task", "Исправить ошибку на основе анализа."),
                        "description": new_task_data.get("description", "Не удалось получить детальное описание от EvaluatorAgent.")
                    }
                    tasks_to_run.append(new_task)
                    print(f"\n\033[1;34m>>> НОВАЯ ЗАДАЧА ДОБАВЛЕНА В ПЛАН: Шаг {new_task['step']}. {new_task['task']}\033[0m")
                except json.JSONDecodeError:
                    logging.error("EvaluatorAgent вернул некорректный JSON: %s", new_task_json_str)
                    print("\n\033[1;31m>>> ОШИБКА: EvaluatorAgent не смог сформировать задачу. Цикл отладки прерван.\033[0m")

            i += 1
            # --- КОНЕЦ НОВОЙ ЛОГИКИ ---

        print(f"\n{'='*20} Цель достигнута! {'='*20}")
        print("Итоговые результаты:")
        for res in results:
            print(f"- Шаг {res['step']}: {res['task']} -> {res['result'][:150]}...")


    def _create_briefing(self, goal: str, plan: List[Dict[str, Any]], current_task: Dict[str, Any], previous_results: List[Dict[str, Any]]) -> str:
        """Создает полный контекст (брифинг) для агента перед выполнением задачи."""
        
        briefing = f"КОНТЕКСТ ВЫПОЛНЕНИЯ ЗАДАЧИ\n"
        briefing += "-" * 30 + "\n"
        briefing += f"**Глобальная цель:** {goal}\n\n"
        briefing += f"**Весь план:**\n"
        for step in plan:
            marker = "-->" if step == current_task else "   "
            briefing += f"{marker} Шаг {step['step']}: {step['task']} (Исполнитель: {step['assignee']})\n"
        
        briefing += f"\n**Текущая задача для тебя:**\n- {current_task['task']}\n- {current_task['description']}\n\n"

        if previous_results:
            briefing += "**Результаты предыдущих шагов:**\n"
            for res in previous_results:
                briefing += f"- Шаг {res['step']} ({res['task']}):\n"
                briefing += f"  Результат: {res['result']}\n"
        else:
            briefing += "**Это первый шаг, предыдущих результатов нет.**\n"
        
        briefing += "-" * 30 + "\n"
        briefing += "Пожалуйста, выполни свою задачу, используя доступные тебе инструменты. Будь краток и точен в своих действиях."
        
        return briefing 

    def _create_evaluator_briefing(self, failed_test_result: str, last_briefing: str) -> str:
        """Создает системный промпт для EvaluatorAgent."""
        return f"""
Ты — 'EvaluatorAgent', старший AI-разработчик и эксперт по отладке.
Твоя единственная задача — анализировать проваленные тесты и создавать четкие, выполнимые задачи для 'CodingAgent'.

# КОНТЕКСТ ПРОВАЛЕННОЙ ЗАДАЧИ:
Ниже приведен полный контекст, который был у агента, когда тесты провалились.
Изучи его, чтобы понять общую цель и предыдущие шаги.
---
{last_briefing}
---

# РЕЗУЛЬТАТ ПРОВАЛЕННЫХ ТЕСТОВ:
А вот лог ошибки, который ты должен проанализировать.
---
{failed_test_result}
---

# ТВОЯ ЗАДАЧА:
1.  **Проанализируй** лог ошибки и контекст. Определи наиболее вероятную причину сбоя.
2.  **Сформулируй** новую, одну-единственную задачу для `CodingAgent`.
    -   Задача должна быть конкретной (например, "Исправить импорт в файле X" или "Изменить логику функции Y в файле Z, чтобы она возвращала тип int").
    -   В описании задачи дай краткую подсказку, на что обратить внимание.
3.  **Верни результат** СТРОГО в формате JSON-объекта. Никакого лишнего текста.

Пример твоего идеального ответа:
```json
{{
    "task": "Исправить путь импорта в файле tests/test_tools.py.",
    "description": "Тесты упали с ошибкой 'ModuleNotFoundError'. Похоже, в файле 'tests/test_tools.py' используется некорректный импорт 'from tools import ...'. Его нужно заменить на правильный, абсолютный импорт 'from app.agents.tools import ...'."
}}
```
""" 
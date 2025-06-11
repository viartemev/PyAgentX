"""
Orchestrator module that manages the execution of a task plan.
"""
import logging
import json
from typing import Dict, Any, List, Tuple, Optional
from app.agents.agent import Agent
from app.agents.roles.task_decomposer import TaskDecomposer

class Orchestrator:
    """
    The Orchestrator manages the entire process: from goal decomposition to task execution by agents.
    """
    def __init__(self, task_decomposer: TaskDecomposer, worker_agents: Dict[str, Agent]):
        self.task_decomposer = task_decomposer
        self.worker_agents = worker_agents
        self.execution_history: List[Dict[str, Any]] = []

    def _get_briefing(self, current_task: Dict[str, Any], plan: List[Dict[str, Any]], goal: str) -> str:
        """Creates a full context (briefing) for an agent before task execution."""
        
        history_log = ""
        if not self.execution_history:
            history_log = "This is the first step, there are no previous results.\n"
        else:
            history_log += "**Execution History:**\n"
            for record in self.execution_history:
                history_log += f"- **Step {record['step']} ({record['assignee']})**: {record['task']}\n"
                history_log += f"  - **Result**: {record['result']}\n"

        plan_log = ""
        for step in plan:
            marker = "-->" if step['step'] == current_task['step'] else "   "
            plan_log += f"{marker} Step {step['step']}: {step['task']} (Assignee: {step['assignee']})\n"

        briefing = f"""
TASK CONTEXT
---------------------------------
**Global Goal:** {goal}

**ENTIRE TASK PLAN:**
{plan_log}
**EXECUTION HISTORY:**
{history_log}
---------------------------------
**YOUR CURRENT TASK (Step {current_task['step']}):**

**Task:** {current_task['task']}
**Description:** {current_task['description']}

Analyze all the provided information, especially the results of previous steps, and execute your task.
"""
        return briefing

    def _perform_code_review(self, task: Dict[str, Any], plan: List[Dict[str, Any]], goal: str) -> Tuple[bool, Optional[str]]:
        """Performs the Code Review step."""
        logging.info("--- Starting Code Review ---")
        reviewer_agent = self.worker_agents.get("ReviewerAgent")
        if not reviewer_agent:
            logging.error("ReviewerAgent not found!")
            return False, "ReviewerAgent was not initialized."

        briefing = self._get_briefing(task, plan, goal)
        review_result = reviewer_agent.execute_task(briefing)

        logging.info(f"Code Review Result: {review_result}")

        if "LGTM" in review_result.upper():
            logging.info("--- Code Review Passed Successfully ---")
            return True, None
        else:
            logging.warning("--- Code Review identified issues ---")
            return False, review_result

    def run(self, goal: str):
        self.execution_history = []
        plan = self._get_plan(goal)
        current_step_index = 0

        while current_step_index < len(plan):
            task = plan[current_step_index]
            logging.info(f"--- Executing Step {task['step']}: {task['task']} ---")

            if task.get("assignee") == "ReviewerAgent":
                review_passed, suggestions = self._perform_code_review(task, plan, goal)
                
                history_record = {
                    "step": task['step'],
                    "task": task['task'],
                    "assignee": task['assignee'],
                }

                if review_passed:
                    history_record["result"] = "Success (LGTM)."
                    self.execution_history.append(history_record)
                    current_step_index += 1
                    continue
                else:
                    history_record["result"] = f"Failed. Feedback: {suggestions}"
                    self.execution_history.append(history_record)
                    
                    logging.warning("Review failed. Creating a task for correction...")
                    new_task_description = (
                        "The reviewer agent found issues in the code you wrote. "
                        f"Here are the comments: '{suggestions}'. "
                        "Your task is to fix the code according to these recommendations."
                    )
                    
                    new_task = {
                        "step": float(task["step"]) + 0.1,
                        "assignee": "CodingAgent",
                        "task": "Fix code based on Code Review feedback.",
                        "description": new_task_description,
                    }
                    
                    plan.insert(current_step_index + 1, new_task)
                    logging.info(f"A new task has been added to the plan: {new_task}")
                    
                    current_step_index += 1
                    continue

            assignee_name = task.get("assignee", "DefaultAgent")
            agent = self.worker_agents.get(assignee_name)
            
            if not agent:
                logging.error(f"Agent {assignee_name} not found! Skipping step.")
                result = f"Skipped (agent '{assignee_name}' not found)."
            else:
                briefing = self._get_briefing(task, plan, goal)
                result = agent.execute_task(briefing)

            logging.info(f"Result of step {task['step']}: {result}")

            if "reached iteration limit" in result:
                logging.critical(f"Agent {assignee_name} failed to complete task {task['step']} and reached the iteration limit. Execution aborted.")
                print(f"CRITICAL ERROR: Agent {assignee_name} failed the task. Check agent_activity.log for details.")
                return
            
            history_record = {
                "step": task['step'],
                "task": task['task'],
                "assignee": task.get("assignee", "DefaultAgent"),
                "result": result
            }
            self.execution_history.append(history_record)

            if agent and agent.name == "TestingAgent" and "FAIL" in result.upper():
                logging.error("Tests failed! Initiating fix process.")
                evaluator_agent = self.worker_agents.get("EvaluatorAgent")
                if not evaluator_agent:
                    logging.error("EvaluatorAgent not found, cannot analyze the error.")
                    current_step_index += 1
                    continue

                evaluator_briefing = (
                    "The tests failed. Analyze the following log and formulate a task to fix it.\n\n"
                    "ERROR LOG:\n"
                    f"{result}"
                )
                
                fix_task_description = evaluator_agent.execute_task(evaluator_briefing)
                logging.info(f"EvaluatorAgent suggested the following task: {fix_task_description}")

                new_task = {
                    "step": float(task["step"]) + 0.1,
                    "assignee": "CodingAgent",
                    "task": "Fix code based on failed tests.",
                    "description": fix_task_description,
                }
                
                plan.insert(current_step_index + 1, new_task)
                logging.info(f"A new fix task has been added to the plan: {new_task}")

            current_step_index += 1

        logging.info("All tasks completed. Goal achieved.")

    def _get_plan(self, goal: str) -> List[Dict[str, Any]]:
        """Gets the plan from the TaskDecomposer."""
        logging.info("Getting plan from TaskDecomposer...")
        plan = self.task_decomposer.get_plan(goal)
        if not plan:
            logging.error("Failed to create a plan. Orchestrator is stopping.")
            return []
        
        logging.info("Plan successfully received.")
        print("\nThe following plan has been created:")
        for step in plan:
            print(f"- Step {step['step']}: {step['task']} (Assignee: {step['assignee']})")
        return plan

    # def _create_evaluator_briefing(self, failed_test_result: str, last_briefing: str) -> str:
    #     """Создает системный промпт для EvaluatorAgent."""
    #     return f"""
    # ... (здесь был промпт)
    # """ 
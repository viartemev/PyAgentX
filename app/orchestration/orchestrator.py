"""
This module defines the Orchestrator, the central brain of the multi-agent team.
It creates a plan and manages its execution by delegating tasks to specialized agents.
"""
import json
import logging
from typing import Dict, Any, List

from openai import OpenAI
from app.factory.agent_factory import AgentFactory
from app.agents.roles.standard_roles import ALL_ROLES, PLANNER_AGENT, EVALUATOR_AGENT

# Updated prompt to ask for multiple plans (Tree-of-Thoughts)
PLANNER_PROMPT_TEMPLATE = """
You are a master planner for a team of AI agents. Your job is to create THREE DISTINCT step-by-step plans to accomplish the user's goal.

**User's Goal:**
"{user_goal}"

**Available Team of Specialists:**
{agents_description}

Based on the goal, create a JSON object with a key "plans", containing a list of THREE different plans. Each plan is a JSON array of tasks. Each task must have:
- `step`: An integer for the step number (e.g., 1, 2, 3).
- `agent`: The name of the single most appropriate agent from the available team.
- `task`: A clear and specific instruction for the agent.

**Important Rules:**
- The three plans should represent different strategies to achieve the goal.
- Your entire response MUST be a single, valid JSON object like: `{{"plans": [[...plan1...], [...plan2...], [...plan3...]]}}`

**Example:**
{{
  "plans": [
    [
      {{"step": 1, "agent": "WebSearchExpert", "task": "Find official pytest docs."}}
    ],
    [
      {{"step": 1, "agent": "FileSystemExpert", "task": "Check existing test files for pytest usage examples."}}
    ],
    [
      {{"step": 1, "agent": "WebSearchExpert", "task": "Search for tutorials on how to use pytest with FastAPI."}}
    ]
  ]
}}
"""

EVALUATOR_PROMPT_TEMPLATE = """
You are a meticulous and rational Evaluator. Your task is to analyze a list of proposed plans and select the single best one to achieve the user's goal.

**User's Goal:**
"{user_goal}"

**Proposed Plans:**
{plans_json_string}

**Evaluation Criteria:**
1.  **Efficiency:** Which plan is likely to achieve the goal in the fewest steps?
2.  **Robustness:** Which plan is least likely to fail or run into errors?
3.  **Clarity:** Which plan is the most logical and straightforward?

Based on your analysis, respond with a JSON object containing the index (starting from 0) of the best plan.

**Example:**
{{
  "best_plan_index": 1
}}
"""

class Orchestrator:
    """
    Creates multiple plans, evaluates them, and manages the execution of the best one.
    """
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.agent_factory = AgentFactory()
        self.api_key = api_key
        self.execution_history: List[Dict[str, Any]] = []

    def _get_agents_description(self) -> str:
        """Creates a formatted string describing the available worker agents."""
        descriptions = []
        for name, config in ALL_ROLES.items():
            descriptions.append(f"- Agent: {name}\n  - Role: {config['role']}\n  - Best for: {config['goal']}")
        return "\n".join(descriptions)

    def _create_plan(self, user_goal: str) -> List[List[Dict[str, Any]]]:
        """
        Uses the PlannerAgent's logic to create multiple distinct plans.
        """
        logging.info(f"Orchestrator is creating multiple plans for the goal: '{user_goal}'")
        agents_description = self._get_agents_description()
        prompt = PLANNER_PROMPT_TEMPLATE.format(
            user_goal=user_goal,
            agents_description=agents_description
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            # We expect a dict with a "plans" key, which is a list of lists
            plan_variants = json.loads(response.choices[0].message.content or "{}").get("plans", [])
            logging.info(f"Planner proposed {len(plan_variants)} plans.")
            return plan_variants
        except Exception as e:
            logging.error(f"Failed to create plans: {e}", exc_info=True)
            return []

    def _evaluate_and_select_plan(self, user_goal: str, plans: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Uses the Evaluator's logic to select the best plan."""
        if not plans:
            return []
        if len(plans) == 1:
            logging.info("Only one plan was generated, selecting it by default.")
            return plans[0]

        logging.info("Evaluating plans to select the best one...")
        prompt = EVALUATOR_PROMPT_TEMPLATE.format(
            user_goal=user_goal,
            plans_json_string=json.dumps(plans, indent=2)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            choice = json.loads(response.choices[0].message.content or "{}")
            best_plan_index = choice.get("best_plan_index", 0)

            if 0 <= best_plan_index < len(plans):
                logging.info(f"Evaluator selected plan #{best_plan_index + 1}.")
                return plans[best_plan_index]
            else:
                logging.warning("Evaluator returned an invalid index, defaulting to the first plan.")
                return plans[0]
        except Exception as e:
            logging.error(f"Failed to evaluate plans: {e}. Defaulting to the first plan.", exc_info=True)
            return plans[0]

    def _create_briefing(self, goal: str, plan: List[Dict[str, Any]], current_task: Dict[str, Any]) -> str:
        """Creates a detailed context (briefing) for an agent."""
        history_log = ""
        if not self.execution_history:
            history_log = "This is the first step. No execution history yet."
        else:
            history_log = "Here are the results from the previous steps:\n"
            for record in self.execution_history:
                history_log += (
                    f"Step {record['step']} ({record['agent']}) completed.\n"
                    f"Task: {record['task']}\n"
                    f"Result: {record['result']}\n\n"
                )

        return (
            f"**Overall Goal:** {goal}\n\n"
            f"**Full Plan:** {json.dumps(plan, indent=2)}\n\n"
            f"**Execution History:**\n{history_log}\n"
            f"-----------------------------------\n"
            f"**Your Current Task (Step {current_task['step']}):**\n"
            f"Your task is: \"{current_task['task']}\".\n"
            f"Analyze the goal, plan, and history, then execute your task to produce the required result."
        )

    def run(self, user_goal: str) -> str:
        """
        Runs the full orchestration process: generate multiple plans, evaluate them, and execute the best one.
        """
        self.execution_history = []
        
        # 1. Create multiple plan variants
        plan_variants = self._create_plan(user_goal)
        if not plan_variants:
            return "I'm sorry, I couldn't create any plans to address your request. Please try rephrasing it."

        # 2. Evaluate and select the best plan
        best_plan = self._evaluate_and_select_plan(user_goal, plan_variants)
        if not best_plan:
             return "I'm sorry, I couldn't select a valid plan to execute. Please try again."

        print(f"\033[95mOrchestrator's Selected Plan:\033[0m\n{json.dumps(best_plan, indent=2)}")

        # 3. Execute the selected plan step-by-step
        for task in best_plan:
            agent_name = task.get("agent")
            task_description = task.get("task")
            
            if not agent_name or not task_description or agent_name not in ALL_ROLES:
                logging.warning(f"Skipping invalid task in plan: {task}")
                continue

            logging.info(f"--- Executing Step {task['step']}: {task_description} (Agent: {agent_name}) ---")
            
            # Create the specialist agent
            agent_config = ALL_ROLES[agent_name]
            specialist_agent = self.agent_factory.create_agent(agent_config, self.api_key, self.model)

            # Create the briefing for the agent
            briefing = self._create_briefing(user_goal, best_plan, task)
            
            # Execute the task
            result = specialist_agent.execute_task(briefing)

            # Save the result to history
            self.execution_history.append({
                "step": task['step'],
                "agent": agent_name,
                "task": task_description,
                "result": result
            })

            logging.info(f"--- Step {task['step']} Result: {result} ---")

        # Return the result of the final step
        final_result = self.execution_history[-1]["result"] if self.execution_history else "The plan was executed, but there is no final result."
        return final_result
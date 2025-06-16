"""
This module defines the Orchestrator, the central brain of the multi-agent team.
It creates a plan and manages its execution by delegating tasks to specialized agents.
"""
import json
import logging
from typing import Dict, Any, List

from openai import OpenAI
from app.factory.agent_factory import AgentFactory
from app.agents.roles.standard_roles import ALL_ROLES, PLANNER_AGENT

# Updated prompt to ask for a multi-step plan
PLANNER_PROMPT_TEMPLATE = """
You are a master planner for a team of AI agents. Your job is to create a step-by-step plan to accomplish the user's goal.

**User's Goal:**
"{user_goal}"

**Available Team of Specialists:**
{agents_description}

Based on the goal, create a JSON array of tasks. Each task must have:
- `step`: An integer for the step number (e.g., 1, 2, 3).
- `agent`: The name of the single most appropriate agent from the available team to perform this step.
- `task`: A clear and specific instruction for what the chosen agent needs to do in this step.

**Important Rules:**
- The plan should be logical and sequential. The result of one step can be used by the next.
- Choose agents whose roles best fit the task for each step.
- Your entire response MUST be a single, valid JSON array.

**Example:**
[
  {{
    "step": 1,
    "agent": "WebSearchExpert",
    "task": "Find the official documentation for the 'pytest' library and provide a summary of its main purpose."
  }},
  {{
    "step": 2,
    "agent": "FileSystemExpert",
    "task": "Based on the summary from the previous step, list all files in the 'tests/' directory to see how pytest is currently used."
  }}
]
"""

class Orchestrator:
    """
    Creates a plan and manages its execution by a team of specialized agents.
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

    def _create_plan(self, user_goal: str) -> List[Dict[str, Any]]:
        """
        Uses the PlannerAgent's logic to create a multi-step plan.
        """
        logging.info(f"Orchestrator is creating a plan for the goal: '{user_goal}'")
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
            plan = json.loads(response.choices[0].message.content or "[]")
            logging.info(f"Plan created successfully: {plan}")
            return plan
        except Exception as e:
            logging.error(f"Failed to create a plan: {e}", exc_info=True)
            return []

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
        Runs the full orchestration process: plan, then execute step-by-step.
        """
        self.execution_history = []
        
        # 1. Create a plan
        plan = self._create_plan(user_goal)
        if not plan or not isinstance(plan, list):
            return "I'm sorry, I couldn't create a valid plan to address your request. Please try rephrasing it."

        print(f"\033[95mOrchestrator Plan:\033[0m\n{json.dumps(plan, indent=2)}")

        # 2. Execute the plan step-by-step
        for task in plan:
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
            briefing = self._create_briefing(user_goal, plan, task)
            
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
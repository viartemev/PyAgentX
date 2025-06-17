"""
This module defines the Orchestrator.
It is the central brain that creates a plan and executes it by delegating tasks to agents.
"""
import json
import logging
from typing import Dict, Any, List

from openai import OpenAI
from app.factory.agent_factory import AgentFactory
from app.agents.roles.standard_roles import ALL_ROLES

# A new, simplified, and much more direct planner prompt.
# This prompt is designed to generate a single, efficient plan.
PLANNER_PROMPT_TEMPLATE = """
You are an expert project planner. Your job is to create a single, efficient, step-by-step plan to accomplish the user's goal.

**User's Goal:**
"{user_goal}"

**Available Team of Specialists & Their Key Tools:**
- **FileSystemExpert**: Ideal for file operations. Key tool: `write_to_file_tool(path, content)`.
- **CodingAgent**: For writing and modifying Python code. Key tool: `write_to_file_tool(path, content)`.
- **TestingAgent**: For running tests. Key tool: `run_tests_tool(path)`.
- **ReviewerAgent**: For code reviews. Key tool: `read_file_tool(path)`.

**Your Task:**
Create a JSON object with a key "plan" containing a list of tasks. Each task must have:
- `step`: An integer for the step number.
- `agent`: The name of the single most appropriate agent from the team.
- `task`: A clear and specific instruction for that agent.

**CRITICAL RULES:**
1.  **BE EFFICIENT:** Combine related actions. Instead of one step to create a file and another to write to it, create a SINGLE step: "Create the file 'file.txt' with the content '...'" and assign it to the appropriate agent.
2.  **BE PRECISE:** The task description must contain all the necessary information for the agent to act. If the goal is to write to a file, the 'task' must include the content to be written.
3.  **JSON ONLY:** Your entire response MUST be a single, valid JSON object.

**Example:**
Goal: "Create a file named 'hello.txt' and write 'Hello World' in it."

Your output:
{{
  "plan": [
    {{
      "step": 1,
      "agent": "FileSystemExpert",
      "task": "Create the file 'hello.txt' and write the following content into it: Hello World"
    }}
  ]
}}
"""

class Orchestrator:
    """
    Creates a single, efficient plan and manages its execution.
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
        """Creates a single, direct plan to achieve the user's goal."""
        logging.info(f"Orchestrator is creating a plan for the goal: '{user_goal}'")
        
        prompt = PLANNER_PROMPT_TEMPLATE.format(user_goal=user_goal)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content or "{}"
            plan = json.loads(content).get("plan", [])
            logging.info("Planner proposed a plan with %s steps.", len(plan))
            return plan
        except Exception as e:
            logging.error(f"Failed to create a valid plan: {e}", exc_info=True)
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
                    f"- Step {record['step']} ({record['agent']}) completed.\n"
                    f"  Task: {record['task']}\n"
                    f"  Result: {record['result']}\n\n"
                )

        # The agent needs the full context to make good decisions.
        return (
            f"**Overall Goal:** {goal}\n\n"
            f"**Full Plan:**\n{json.dumps(plan, indent=2)}\n\n"
            f"**Execution History:**\n{history_log}\n"
            f"-----------------------------------\n"
            f"**Your Current Task (Step {current_task['step']}):**\n"
            f"Your task is: \"{current_task['task']}\".\n\n"
            f"Analyze the goal, plan, and history, then use your tools to execute your task and produce the required result. "
            f"The 'task' description contains all the information you need."
        )

    def run(self, user_goal: str) -> str:
        """
        Runs the full orchestration process: create a plan and execute it.
        """
        self.execution_history = []
        
        # 1. Create a single, efficient plan
        plan = self._create_plan(user_goal)
        if not plan:
            return "I'm sorry, I couldn't create a plan to address your request. Please try rephrasing it."

        print(f"\033[95mOrchestrator's Plan:\033[0m\n{json.dumps(plan, indent=2, ensure_ascii=False)}")

        # 2. Execute the plan step-by-step
        for task in plan:
            agent_name = task.get("agent")
            task_description = task.get("task")
            
            if not agent_name or not task_description or agent_name not in ALL_ROLES:
                logging.warning(f"Skipping invalid task in plan: {task}")
                continue

            logging.info(f"--- Executing Step {task['step']}: {task_description} (Agent: {agent_name}) ---")
            
            agent_config = ALL_ROLES[agent_name]
            specialist_agent = self.agent_factory.create_agent(agent_config, self.api_key, self.model)

            briefing = self._create_briefing(user_goal, plan, task)
            
            result = specialist_agent.execute_task(briefing)

            self.execution_history.append({
                "step": task['step'],
                "agent": agent_name,
                "task": task_description,
                "result": result
            })

            logging.info(f"--- Step {task['step']} Result: {result} ---")

        final_result = self.execution_history[-1]["result"] if self.execution_history else "The plan was executed, but there is no final result."
        return final_result
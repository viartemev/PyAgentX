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
You are an expert project planner and a senior software architect. Your job is to create a robust, production-ready, step-by-step plan to accomplish the user's goal.

**User's Goal:**
"{user_goal}"

**Project Structure & Quality Standards:**
- **Tool Directory**: All new tools MUST be appended to the existing file `app/agents/tools.py`. Do NOT create new files for tools.
- **Test Directory**: All new tests for tools MUST be appended to the existing file `tests/agents/test_tools.py`. Do NOT create new files for tests.
- **Code Quality**: All generated Python functions MUST include detailed Google-style docstrings explaining their purpose, arguments, and return values.
- **Test Quality**: Tests MUST be written using `pytest`. Each feature MUST have multiple tests covering standard cases, edge cases (e.g., empty strings, null inputs, different data types), and potential failure modes.
- **Mandatory Review**: Any plan that involves writing or modifying code MUST include a final step where the `ReviewerAgent` inspects the newly created files.

**Available Team of Specialists & Their Exact Responsibilities:**
- **CodingAgent**: Writes and modifies Python code. Can ONLY use `write_to_file_tool`, `read_file_tool`, and `list_files_tool`. Assign ALL code and test writing/modification tasks to this agent.
- **TestingAgent**: Runs tests using `pytest`. Can ONLY use `run_tests_tool` and `read_file_tool`. It CANNOT write or modify files. Assign ONLY test execution tasks to this agent.
- **ReviewerAgent**: Performs code reviews. Can ONLY use `read_file_tool`. It CANNOT write, modify, or test code. Assign ONLY file review tasks to this agent.
- **FileSystemExpert**: Handles generic file operations. Can ONLY use `write_to_file_tool`, `read_file_tool`, and `list_files_tool`.

**Your Task:**
Create a JSON object with a key "plan" containing a list of tasks. Each task must have:
- `step`: An integer for the step number.
- `agent`: The name of the single most appropriate agent from the team based on their exact responsibilities.
- `task`: A clear and specific instruction for that agent, including full file paths and complete, high-quality code with docstrings.

**CRITICAL RULES:**
1.  **ADHERE TO STANDARDS:** Your plan MUST follow all project structure and quality standards defined above.
2.  **CORRECT ASSIGNMENT:** Assign tasks ONLY to agents that have the tools and responsibilities to complete them.
3.  **BE PRECISE:** The task description must contain all necessary information, including full code content.
4.  **JSON ONLY:** Your entire response MUST be a single, valid JSON object.

**Example of a HIGH-QUALITY Plan:**
Goal: "Create a Python tool to reverse a string."

Your output:
{{
  "plan": [
    {{
      "step": 1,
      "agent": "CodingAgent",
      "task": "Read the content of 'app/agents/tools.py'. If the 'reverse_string_tool' function does not already exist, append the following code to the end of the file: \\n\\n# ... (existing code) ...\\n\\ndef reverse_string_tool(input_data: dict) -> str:\\n    '''Reverses a string.'''\\n    return input_data.get('text', '')[::-1]\\n\\nreverse_string_tool_def = {{...}} # (full tool definition here)"
    }},
    {{
      "step": 2,
      "agent": "CodingAgent",
      "task": "Read the content of 'tests/agents/test_tools.py'. If tests for 'reverse_string_tool' do not already exist, append new tests to the end of the file."
    }},
    {{
      "step": 3,
      "agent": "TestingAgent",
      "task": "Run all tests in 'tests/test_tools.py' to verify correctness of all tools, including the newly added one."
    }},
    {{
      "step": 4,
      "agent": "ReviewerAgent",
      "task": "Review the final code in 'app/agents/tools.py' and 'tests/agents/test_tools.py' for quality and correctness."
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

            # This briefing is now lean and focused. It does NOT contain the overall goal.
            briefing = ""
            if self.execution_history:
                briefing += (
                    "Here's a summary of what has been done so far:\n"
                    + "\n".join([f"- {record['result']}" for record in self.execution_history])
                )
            briefing += (
                f"\nYour specific, immediate task is: '{task_description}'.\n"
                "Focus ONLY on completing this single task. Do not move on to other tasks. "
                "Once you have completed your task, provide a 'Final Answer' with a summary of what you did."
            )

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
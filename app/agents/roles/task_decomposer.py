"""Task Decomposer Agent."""
from app.agents.agent import Agent
import logging
import json

class TaskDecomposer(Agent):
    """
    An agent specializing in breaking down a high-level goal into a step-by-step plan.
    """
    def __init__(self, name: str, role: str, goal: str, **kwargs):
        # This agent typically does not need RAG, but the option is there.
        kwargs.setdefault('use_rag', False)
        
        super().__init__(
            name=name,
            role=role,
            goal=goal,
            **kwargs,
        )
        self.system_prompt = """
You are an expert project manager. Your task is to break down a high-level user goal into a concise, step-by-step plan.
Each step must be a single, clear action assigned to one of the available roles.

# AVAILABLE ROLES:
- CodingAgent: Writes, modifies, and fixes code.
- TestingAgent: Runs tests and reports results.
- ReviewerAgent: Performs code reviews, checking for quality and adherence to standards.
- EvaluatorAgent: Analyzes test failures and creates bug reports.

# OUTPUT FORMAT:
Your output must be a list of steps in JSON format. Do not include any other text or explanation.

# EXAMPLE:
Goal: "Create a function to add two numbers and test it."

Your output:
```json
[
  {
    "step": 1,
    "assignee": "CodingAgent",
    "task": "Create a new function 'add(a, b)' in 'app/utils/math.py'."
  },
  {
    "step": 2,
    "assignee": "ReviewerAgent",
    "task": "Review the 'add' function in 'app/utils/math.py'."
  },
  {
    "step": 3,
    "assignee": "CodingAgent",
    "task": "Create a new test file 'tests/test_math.py' with tests for the 'add' function."
  },
  {
    "step": 4,
    "assignee": "TestingAgent",
    "task": "Run the tests in 'tests/test_math.py'."
  }
]
```
"""

    def get_plan(self, goal: str) -> list:
        """
        Generates a plan for a given goal.
        Overrides the base 'execute_task' to return a structured plan.
        """
        task_briefing = f"Create a step-by-step plan to achieve the following goal: {goal}"
        
        try:
            logging.info("Requesting plan from OpenAI...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task_briefing},
                ],
                response_format={"type": "json_object"},
            )
            
            response_content = response.choices[0].message.content
            logging.info("Received raw plan: %s", response_content)
            
            # The response is a JSON string, so we need to parse it.
            plan = json.loads(response_content)
            # Sometimes the model returns a dictionary with a "plan" key
            if isinstance(plan, dict) and "plan" in plan:
                return plan["plan"]
            return plan

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from OpenAI response: {e}")
            logging.error(f"Raw response was: {response_content}")
            return [{"step": 1, "assignee": "DefaultAgent", "task": "Failed to create a valid plan due to JSON error."}]
        except Exception as e:
            logging.error(f"An unexpected error occurred while getting the plan: {e}", exc_info=True)
            return [{"step": 1, "assignee": "DefaultAgent", "task": "Failed to create a plan due to an unexpected error."}] 
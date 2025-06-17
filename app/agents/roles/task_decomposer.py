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
Each step must be a single, clear action assigned to an appropriate agent.
Combine simple, related actions into a single, comprehensive step. For example, instead of one step to create a file and another to write to it, create a single step that does both.

# AVAILABLE ROLES & THEIR KEY TOOLS:
- **FileSystemExpert**: Works with files. Key tool: `write_to_file_tool(path, content)`.
- **CodingAgent**: Writes, modifies, and fixes Python code. Key tool: `write_to_file_tool(path, content)`.
- **TestingAgent**: Runs tests. Key tool: `run_tests_tool(path)`.
- **ReviewerAgent**: Performs code reviews. Key tool: `read_file_tool(path)`.

# OUTPUT FORMAT:
Your output MUST be a single JSON object with a single key "plan", which contains a list of steps. Do not include any other text, explanation, or markdown code fences.

# EXAMPLE 1: Simple file operation
Goal: "Create a file named 'hello.txt' and write 'Hello World' in it."

Your output:
{
  "plan": [
    {
      "step": 1,
      "agent": "FileSystemExpert",
      "task": "Create a new file 'hello.txt' with the content 'Hello World'."
    }
  ]
}

# EXAMPLE 2: More complex coding task
Goal: "Create a function to add two numbers and test it."

Your output:
{
  "plan": [
    {
      "step": 1,
      "agent": "CodingAgent",
      "task": "Create a new file 'app/utils/math.py' with an 'add(a, b)' function that returns the sum of two numbers."
    },
    {
      "step": 2,
      "agent": "CodingAgent",
      "task": "Create a new test file 'tests/test_math.py' to test the 'add' function. Include at least one test case."
    },
    {
      "step": 3,
      "agent": "TestingAgent",
      "task": "Run the tests in 'tests/test_math.py'."
    }
  ]
}
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
            )
            
            response_content = response.choices[0].message.content
            logging.info("Received raw plan: %s", response_content)
            
            # The response is a JSON string, so we need to parse it.
            parsed_json = json.loads(response_content)
            
            # The model should return a dict with a "plan" key
            if isinstance(parsed_json, dict) and "plan" in parsed_json:
                return parsed_json["plan"]
            else:
                logging.error("Invalid plan format. 'plan' key not found in response.")
                raise ValueError("Invalid plan format.")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from OpenAI response: {e}")
            logging.error(f"Raw response was: {response_content}")
            return [{"step": 1, "assignee": "DefaultAgent", "task": "Failed to create a valid plan due to JSON error.", "description": "The model returned malformed JSON."}]
        except Exception as e:
            logging.error(f"An unexpected error occurred while getting the plan: {e}", exc_info=True)
            return [{"step": 1, "assignee": "DefaultAgent", "task": "Failed to create a plan due to an unexpected error.", "description": f"An unexpected error occurred: {e}"}] 
"""
Module for decomposing a high-level task into specific subtasks.
"""
import json
import logging
from typing import List, Optional, Dict, Any
from openai import OpenAI

DECOMPOSER_PROMPT = """
Твоя задача - разбить высокоуровневую цель на детальный, последовательный план для AI-агента.
Ответ должен быть ТОЛЬКО JSON-массивом строк без какого-либо другого текста или объяснений.

# Доступные Инструменты Агента:
Агент имеет доступ к следующим функциям, которые он может использовать для выполнения шагов:
- `list_files_tool(path: str)`: Показывает содержимое директории.
- `read_file_tool(path: str)`: Читает содержимое файла.
- `edit_file_tool(path: str, mode: str, ...)`: Редактирует файл. Режимы: 'append' (добавить в конец), 'replace' (заменить фрагмент), 'overwrite' (перезаписать).
- `delete_file_tool(path: str)`: Удаляет файл.

# Правила Создания Плана:
1.  **Конкретика**: Каждый шаг должен быть одной конкретной, осмысленной операцией. Думай о том, как бы ты сам решал эту задачу, используя доступные инструменты.
2.  **Эффективность**: Не создавай лишних шагов. Например, не нужно отдельно "проверять существование файла", если следующий шаг - его чтение. Инструмент `read_file_tool` сам сообщит об ошибке, если файла нет.
3.  **Никаких фиктивных шагов**: Не создавай шаги, для которых нет инструментов. Например, "закрыть файл", "проверить синтаксис" или "запустить тесты" - это плохие шаги, так как у агента нет для них инструментов.
4.  **Целостность**: Думай о всем процессе. Если один шаг генерирует код, следующий шаг должен использовать этот код (например, сохранить его в файл).

# Пример:
### Цель:
"Проанализируй файл `main.py`, предложи улучшение и запиши новую версию в `main_v2.py`."

### Хороший план (твой результат должен быть в таком формате):
[
  "Прочитать содержимое файла `main.py` для анализа.",
  "Проанализировать прочитанный код и сгенерировать новую, улучшенную версию кода.",
  "Записать сгенерированный улучшенный код в новый файл `main_v2.py`."
]


# Твоя Задача:
### Цель:
"{main_goal}"

### План (только JSON):
"""

class TaskDecomposer:
    """
    Decomposes the main task into a list of subtasks using an LLM.
    """
    def __init__(self, api_key: str, model: str = "gpt-4-turbo"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _parse_llm_response(self, content: Optional[str]) -> List[str]:
        """Parses the LLM response more flexibly."""
        if not content:
            logging.warning("LLM returned an empty response during decomposition.")
            return []
        
        try:
            # Try to load as a full JSON
            parsed_json = json.loads(content)
            
            if isinstance(parsed_json, list):
                return [str(item) for item in parsed_json]
            
            # If it's a dictionary, look for a key containing a list
            if isinstance(parsed_json, dict):
                for key, value in parsed_json.items():
                    if isinstance(value, list):
                        logging.info("Found plan in key '%s'", key)
                        return [str(item) for item in value]
            
            logging.warning("LLM response is not a list and does not contain a list. Response: %s", content)
            return []

        except json.JSONDecodeError:
            # Sometimes the LLM returns a raw list without quotes, try to "fix" it
            logging.warning("Failed to parse JSON. Response: %s", content)
            # This is a very simplified attempt to extract strings, may not always work
            cleaned_content = content.strip().replace("`", "")
            if cleaned_content.startswith('[') and cleaned_content.endswith(']'):
                try:
                    return json.loads(cleaned_content)
                except json.JSONDecodeError:
                    pass
            logging.error("Failed to extract plan from LLM response.")
            return []

    def generate_plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Generates a step-by-step plan to achieve a goal, assigning executors.
        """
        # Updated prompt with role descriptions and requirement to assign an executor
        system_prompt = """
You are an elite AI planner specializing in decomposing complex IT tasks for a team of AI agents.
Your task is to break down a high-level goal into a detailed, sequential plan in JSON format.

# PROJECT STRUCTURE (CRITICALLY IMPORTANT):
-   `app/`: All application logic is here.
    -   `app/agents/`: The agents' code.
    -   `app/agents/tools.py`: **File with tools used by the agents.**
-   `tests/`: All tests are here.
    -   `tests/test_tools.py`: **Tests for the tools from `app/agents/tools.py`**

# AVAILABLE AGENT TEAM:
1.  **CodingAgent**:
    -   **Specialization**: Writing, reading, and modifying code.
    -   **Tools**: `list_files`, `read_file`, `edit_file`.
2.  **TestingAgent**:
    -   **Specialization**: Testing code.
    -   **Tools**: `read_file`, `run_tests`.
3.  **ReviewerAgent**:
    -   **Specialization**: Checking code quality, finding bugs and inconsistencies.
    -   **Tools**: `read_file`.
4.  **EvaluatorAgent**:
    -   **Specialization**: Analyzing errors and creating bug reports.
    -   **Tools**: `read_file`.
    -   **IMPORTANT**: This agent is used by the Orchestrator automatically when tests fail. You do not need to assign it tasks in the initial plan.
5.  **DefaultAgent**:
    -   **Specialization**: General tasks and analysis.
    -   **Tools**: `list_files`, `read_file`.

# RULES FOR CREATING THE PLAN:
1.  **Code Review for ALL code**: Immediately after EVERY step in which `CodingAgent` writes or modifies any code (be it main code, tests, documentation, etc.), a "Conduct Code Review" step assigned to `ReviewerAgent` MUST follow.
2.  **Full Paths**: ALWAYS use full relative paths from the project root for any files. For example: `app/agents/tools.py`, `tests/test_tools.py`.
3.  **Specificity**: Formulate tasks as specifically as possible. Instead of "write a function", write "add function X to file Y".
4.  **Logic Before Tests**: The plan must first contain a step for writing or changing the **complete logic** of a function, and only then a step for writing tests.
5.  **Focused Testing**: The testing step must specify a particular test file. In the task `description`, you MUST include an example of how to call the tool, for example: `Using 'run_tests', call it like this: run_tests_tool({'path': 'tests/test_tools.py'})`.
6.  **Assignment**: For each step, specify the `assignee` (`CodingAgent`, `TestingAgent`, `DefaultAgent`).
7.  **Format**: The output must be STRICTLY in the format of a JSON array of objects.

# EXAMPLE:

**Goal**: "Add a `multiply(a, b)` function to `tools.py` and cover it with tests."

**Result (JSON):**
```json
[
    {
        "step": 1,
        "assignee": "CodingAgent",
        "task": "Add 'multiply_tool' function to 'app/agents/tools.py'.",
        "description": "Open the file 'app/agents/tools.py' and add a new Python function 'multiply' that takes two numerical arguments and returns their product. Use 'append' mode."
    },
    {
        "step": 2,
        "assignee": "ReviewerAgent",
        "task": "Conduct Code Review for 'multiply_tool' function in 'app/agents/tools.py'.",
        "description": "Check the code for compliance with quality standards."
    },
    {
        "step": 3,
        "assignee": "CodingAgent",
        "task": "Create 'tests/test_tools.py' file with tests for 'multiply_tool'.",
        "description": "In 'tests/test_tools.py', write a 'test_multiply' test using pytest that checks the correctness of the 'multiply' function."
    },
    {
        "step": 4,
        "assignee": "ReviewerAgent",
        "task": "Conduct Code Review for 'tests/test_tools.py' file.",
        "description": "Check the test code for completeness, correctness, and style."
    },
    {
        "step": 5,
        "assignee": "TestingAgent",
        "task": "Run tests for 'tests/test_tools.py' file.",
        "description": "Using the 'run_tests' tool, call it with the argument {'path': 'tests/test_tools.py'} to verify the work done."
    }
]
```
"""
        user_prompt = f"My goal is: \"{goal}\". Please create a plan."

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            )
            
            plan_str = response.choices[0].message.content
            
            if plan_str.strip().startswith("```json"):
                plan_str = plan_str.strip()[7:-3].strip()

            plan = json.loads(plan_str)
            
            for step in plan:
                if 'assignee' not in step:
                    raise ValueError(f"Step {step.get('step')} in the plan is missing the required 'assignee' field")

            return plan

        except Exception as e:
            logging.error("An error occurred during task decomposition: %s", e, exc_info=True)
            return [] 
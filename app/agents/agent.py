"""
Этот модуль определяет основного AI-агента, его логику и цикл взаимодействия с пользователем.
"""
from typing import Callable, Optional, List, Dict, Any
import os
import json
import logging
import re
from openai import OpenAI, APIError
from openai.types.chat import ChatCompletionMessage
from dotenv import load_dotenv
import inspect

from app.rag.retriever import KnowledgeRetriever
from app.agents.tools import (
    read_file_tool, read_file_tool_def,
    list_files_tool, list_files_tool_def,
    write_to_file_tool, write_to_file_tool_def,
)
from app.agents.web_search_tool import web_search_tool, web_search_tool_def
from app.agents.memory_tool import save_memory_tool, save_memory_tool_def
from app.memory.memory_manager import MemoryManager
from app.safety.custom_guardrails import CustomGuardrailManager

def _extract_json_from_response(response_text: str) -> Optional[str]:
    """
    Extracts a JSON object from a string that might be wrapped in markdown code blocks.
    """
    # Regex to find JSON wrapped in ```json ... ``` or just ``` ... ```
    match = re.search(r'```(json\s*)?(?P<json>\{.*?\})```', response_text, re.DOTALL)
    if match:
        return match.group('json')
    
    # If no markdown block is found, assume the whole string is a JSON object
    # and try to find the start of a JSON object.
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    if first_brace != -1 and last_brace != -1:
        return response_text[first_brace:last_brace+1]
        
    return None

class ToolExecutionError(Exception):
    """Custom exception for errors during tool execution."""
    pass

# Новый системный промпт, основанный на ReAct (Reason + Act)
REACT_SYSTEM_PROMPT = """
You are a smart, autonomous AI agent. Your name is {agent_name}, and your role is {agent_role}.

Your current, specific task is described below. Focus ONLY on this task.
---
{task_description}
---

You operate in a loop of Thought, Action, and Observation.
At each step, you MUST respond in a specific JSON format. Your entire response must be a single JSON object.

1.  **Thought**: First, think about your plan to accomplish your assigned task. Analyze the task description and the previous steps. Describe your reasoning for the current action. This is a mandatory field.

2.  **Action** or **Answer**: Based on your thought, you must choose ONE of the following:
    a. `action`: An object representing the tool to use. It must contain:
       - `name`: The name of the tool to execute.
       - `input`: An object with the parameters for the tool.
    b. `answer`: A final, comprehensive summary of what you did to complete YOUR ASSIGNED TASK. Use this ONLY when your specific task is fully complete.

# AVAILABLE TOOLS:
You have access to the following tools. Use them to gather information and perform actions.

{tools_description}

# LONG-TERM MEMORY:
Before you begin, here are some facts you have previously saved to your long-term memory.
Use them to inform your decisions, but do not state them unless relevant.

{memory_context}

# EXAMPLE OF A SINGLE STEP:

```json
{{
  "thought": "I need to understand the project structure first. I'll list the files in the current directory.",
  "action": {{
    "name": "list_files_tool",
    "input": {{
      "path": "."
    }}
  }}
}}
```

# IMPORTANT RULES:
- Your ENTIRE output MUST be a single, valid JSON object. Do not add any text before or after the JSON.
- You must choose either `action` or `answer`, not both.
- The `thought` field is always required.
- Once you have completed your specific task, you MUST use the 'answer' field to finish your work.
"""

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Agent:
    """
    ИИ-агент, который может использовать инструменты (функции) для ответов на вопросы.
    """
    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        api_key: str,
        model: str = "o4-mini",
        max_iterations: int = 10,
        use_rag: bool = False,
        rag_config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.role = role
        self.goal = goal
        self.model = model
        self.client = OpenAI(api_key=api_key)
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.max_iterations = max_iterations
        self.system_prompt_template = REACT_SYSTEM_PROMPT

        # Initialize Memory Manager
        self.memory_manager = MemoryManager()
        # Initialize Guardrails
        self.guardrail_manager = CustomGuardrailManager(api_key=api_key)

        # Tools are now added by the AgentFactory based on the role's configuration.
        # The base agent starts with an empty toolset.
        
        # RAG specific attributes
        self.use_rag = use_rag
        self.rag_config = rag_config or {}
        self.retriever: Optional[KnowledgeRetriever] = None
        if self.use_rag:
            try:
                self.retriever = KnowledgeRetriever()
            except FileNotFoundError as e:
                logging.warning(
                    f"Agent '{self.name}' was configured to use RAG, "
                    f"but the knowledge base is not found. RAG will be disabled. "
                    f"Error: {e}"
                )
                self.use_rag = False

    def add_tool(self, tool_func: Callable, tool_definition: Dict[str, Any]):
        """Добавляет инструмент и его определение."""
        tool_name = tool_func.__name__
        self.tools[tool_name] = tool_func
        self.tool_definitions.append(tool_definition)
    
    def _get_memory_context(self) -> str:
        """Retrieves relevant facts from long-term memory."""
        recent_facts = self.memory_manager.get_recent_facts(limit=10)
        if not recent_facts:
            return "No relevant facts found in memory."
        
        formatted_facts = "\n".join([f"- {fact}" for fact in recent_facts])
        return f"--- RELEVANT FACTS ---\n{formatted_facts}\n----------------------"

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Any:
        """Выполняет указанный инструмент с аргументами."""
        if tool_name in self.tools:
            try:
                # Наши функции-инструменты ожидают один словарь в качестве аргумента
                result = self.tools[tool_name](tool_args)
                return result
            except Exception as e:
                logging.error("Ошибка при выполнении инструмента '%s': %s", tool_name, e, exc_info=True)
                # Выбрасываем кастомное исключение, чтобы его можно было поймать выше
                raise ToolExecutionError(f"Error: Failed to execute tool '{tool_name}'. Reason: {e}")
        else:
            logging.warning("Попытка вызова неизвестного инструмента: '%s'", tool_name)
            raise ToolExecutionError(f"Error: Tool '{tool_name}' not found.")

    def _get_tools_description(self) -> str:
        """Generates a description of available tools for the system prompt."""
        if not self.tool_definitions:
            return "No tools available."
        
        desc = []
        for tool in self.tool_definitions:
            func_desc = tool.get('function', {})
            params_desc = func_desc.get('parameters', {}).get('properties', {})
            
            # Описание параметров
            params_str = ", ".join([f"{name}: {info.get('type')}" for name, info in params_desc.items()])
            
            desc.append(
                f"- `{func_desc.get('name', 'N/A')}({params_str})`: {func_desc.get('description', 'No description.')}"
            )
        return "\n".join(desc)

    def _get_model_response(self) -> str:
        """Получает и возвращает ответ от модели."""
        logging.info(f"Sending request to OpenAI with {len(self.conversation_history)} messages.")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                # tools=self.get_openai_tools(),
                # tool_choice="auto",
                temperature=0.1,
                top_p=0.1,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content or ""
        except APIError as e:
            logging.error("OpenAI API error: %s", e)
            return f'{{"thought": "Encountered an API error. I should try again.", "action": {{"name": "wait", "input": {{"seconds": 2}}}}}}'

    def execute_task(self, briefing: str) -> str:
        """
        Основной цикл выполнения задачи агентом.
        """
        # 1. Формируем системный промпт
        tools_description = self._get_tools_description()
        memory_context = self._get_memory_context()

        # The 'briefing' from the orchestrator is the definitive task.
        system_prompt = self.system_prompt_template.format(
            agent_name=self.name,
            agent_role=self.role,
            task_description=briefing,  # The briefing is the task
            tools_description=tools_description,
            memory_context=memory_context
        )

        self.conversation_history = [{"role": "system", "content": system_prompt}]
        # A simple message to kick off the process.
        self.conversation_history.append({"role": "user", "content": "Begin your task."})

        # 2. Основной цикл Reason-Act
        for i in range(self.max_iterations):
            logging.info(f"--- Iteration {i+1}/{self.max_iterations} ---")
            
            # 3. Получаем ответ от модели
            response_text = self._get_model_response()
            
            # --- Guardrails Input Check ---
            # guardrail_verdict = self.guardrail_manager.check_input(response_text)
            # if guardrail_verdict:
            #     final_answer = f"Input check failed: {guardrail_verdict}"
            #     logging.warning(final_answer)
            #     self.conversation_history.append({"role": "assistant", "content": final_answer})
            #     return final_answer
            
            # 4. Парсим JSON из ответа
            json_response = None
            try:
                # Используем новую функцию для извлечения JSON
                cleaned_response = _extract_json_from_response(response_text)
                if not cleaned_response:
                    raise json.JSONDecodeError("No JSON object found in response", response_text, 0)
                
                json_response = json.loads(cleaned_response)
                
                self.conversation_history.append({"role": "assistant", "content": cleaned_response})

            except json.JSONDecodeError:
                error_message = f"Error parsing model response: Expecting value: line 1 column 1 (char 0). Response was: '{response_text}'"
                logging.error(error_message)
                # Добавляем сообщение об ошибке в историю, чтобы модель могла исправиться
                self.conversation_history.append({
                    "role": "user", 
                    "content": f"Your last response was not a valid JSON. Please correct your output to be a single JSON object. Error: {error_message}"
                })
                continue # Переходим к следующей итерации, чтобы модель могла исправиться

            # 5. Извлекаем мысль и действие/ответ
            thought = json_response.get("thought")
            action = json_response.get("action")
            answer = json_response.get("answer")

            logging.info(f"Thought: {thought}")

            if answer:
                logging.info(f"Final Answer: {answer}")
                # --- Guardrails Output Check ---
                # guardrail_verdict = self.guardrail_manager.check_output(answer)
                # if guardrail_verdict:
                #     final_answer = f"Output check failed: {guardrail_verdict}"
                #     logging.warning(final_answer)
                #     return final_answer
                
                return str(answer) # Ensure answer is a string

            if action:
                # The model sometimes returns a string in the 'action' field when it should 
                # be providing a final 'answer'. We'll treat this as a final answer to
                # make the system more robust.
                if not isinstance(action, dict):
                    logging.warning(f"Model provided 'action' as a string instead of an object: '{action}'. Treating as Final Answer.")
                    return str(action)
                    
                tool_name = action.get("name")
                tool_input = action.get("input", {})

                # The model sometimes tries to call 'answer' as a tool. This is incorrect.
                # We'll catch this and treat it as the final answer.
                if tool_name == 'answer':
                    logging.warning("Model incorrectly used 'answer' as a tool name. Treating as Final Answer.")
                    return json.dumps(tool_input) if isinstance(tool_input, dict) else str(tool_input)
                
                if tool_name and isinstance(tool_input, dict):
                    logging.info(f"Action: {tool_name}({tool_input})")
                    try:
                        # 6. Выполняем инструмент
                        tool_result = self._execute_tool(tool_name, tool_input)
                        observation = f"Tool {tool_name} executed successfully. Result:\n{tool_result}"
                    except ToolExecutionError as e:
                        observation = str(e) # Используем сообщение из кастомного исключения
                    
                    logging.info(f"Observation: {observation}")
                    
                    # 7. Добавляем результат в историю для следующего шага
                    self.conversation_history.append({"role": "user", "content": f"Observation: {observation}"})
                else:
                    logging.warning("Invalid action format in model response.")
                    self.conversation_history.append({"role": "user", "content": "Observation: Invalid action format. Please provide a valid 'name' and 'input' for the action."})
            else:
                logging.warning("Model did not provide an 'action' or 'answer'.")
                # Просим модель предоставить либо действие, либо финальный ответ
                self.conversation_history.append({"role": "user", "content": "Observation: You must provide either an 'action' or a final 'answer'."})
        
        final_answer = f"Agent {self.name} reached max iterations ({self.max_iterations}) without a final answer."
        logging.warning(final_answer)
        return final_answer


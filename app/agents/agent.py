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
Your ultimate goal is: {agent_goal}.

You operate in a loop of Thought, Action, and Observation.
At each step, you MUST respond in a specific JSON format. Your entire response must be a single JSON object.

1.  **Thought**: First, think about your plan. Analyze the user's request, your goal, and the previous steps. Describe your reasoning for the current action. This is a mandatory field.

2.  **Action** or **Answer**: Based on your thought, you must choose ONE of the following:
    a. `action`: An object representing the tool to use. It must contain:
       - `name`: The name of the tool to execute.
       - `input`: An object with the parameters for the tool.
    b. `answer`: A final, comprehensive answer to the user's request. Use this ONLY when the task is fully complete.

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
- Think step by step. Your goal is to complete the task, not just use tools.
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

        # Add default tools
        self.add_tool(read_file_tool, read_file_tool_def)
        self.add_tool(list_files_tool, list_files_tool_def)
        self.add_tool(write_to_file_tool, write_to_file_tool_def)
        self.add_tool(web_search_tool, web_search_tool_def)
        self.add_tool(save_memory_tool, save_memory_tool_def)
        
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
    
    def get_openai_tools(self) -> Optional[List[Dict[str, Any]]]:
        """Возвращает список определений инструментов для OpenAI API."""
        if not self.tool_definitions:
            return None
        return self.tool_definitions
    
    def _create_rag_query(self, briefing: str) -> str:
        """
        Creates a focused query for RAG from the detailed briefing.
        Extracts the current task and the last result from history.
        """
        task_match = re.search(r"\*\*YOUR CURRENT TASK \(Step .*\):\*\*\n\n\*\*Task:\*\* (.*)\n\*\*Description:\*\* (.*)", briefing)
        history_match = re.search(r"\*\*EXECUTION HISTORY:\*\*\n(.*)", briefing, re.DOTALL)

        if not task_match:
            return briefing # Fallback to full briefing

        task = task_match.group(1)
        description = task_match.group(2)
        
        query_parts = [f"Task: {task}", f"Description: {description}"]

        if history_match:
            history_str = history_match.group(1).strip()
            # Get the last entry from the history
            last_entry = history_str.split("- **Step")[0].strip()
            if last_entry:
                query_parts.append(f"Context from previous step: {last_entry}")
        
        focused_query = "\n".join(query_parts)
        logging.info(f"Created focused RAG query for {self.name}: '{focused_query}'")
        return focused_query

    def _enrich_with_knowledge(self, query: str) -> str:
        """Enriches a query with context from the knowledge base if RAG is enabled."""
        if not self.use_rag or not self.retriever:
            return ""

        top_k = self.rag_config.get("top_k", 3)
        filters = self.rag_config.get("filters", None)

        logging.info(f"Agent '{self.name}' is retrieving knowledge with top_k={top_k}, filters={filters}")
        retrieved_knowledge = self.retriever.retrieve(query=query, top_k=top_k, filters=filters)

        if not retrieved_knowledge:
            logging.info("No specific internal standards found for this query.")
            return ""

        formatted_knowledge = "\n\n---\n\n".join(
            [f"Source: {chunk['source']}\n\n{chunk['text']}" for chunk in retrieved_knowledge]
        )
        knowledge_context = (
            "Before you begin, consult these internal standards and best practices:"
            f"\n\n--- RELEVANT KNOWLEDGE ---\n{formatted_knowledge}\n--------------------------\n"
        )
        logging.info(f"Knowledge context added for agent '{self.name}'.")
        return knowledge_context

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

    def _get_user_input(self) -> Optional[str]:
        """Обрабатывает получение ввода от пользователя."""
        print("\033[94mВы:\033[0m ", end="")
        try:
            user_input = input()
            return user_input if user_input.strip() else None
        except EOFError:
            return None

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

        system_prompt = self.system_prompt_template.format(
            agent_name=self.name,
            agent_role=self.role,
            agent_goal=self.goal,
            tools_description=tools_description,
            memory_context=memory_context
        )

        self.conversation_history = [{"role": "system", "content": system_prompt}]
        self.conversation_history.append({"role": "user", "content": briefing})

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
                
                return answer

            if action:
                tool_name = action.get("name")
                tool_input = action.get("input", {})
                
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

    def run(self) -> None:
        """Запускает основной цикл общения с агентом в интерактивном режиме."""
        print(f"Запуск агента '{self.name}' в интерактивном режиме. Введите 'exit' для завершения.")
        
        while True:
            try:
                user_input = input("\033[94mВы > \033[0m")
                if user_input.lower() == 'exit':
                    print("Завершение сеанса.")
                    break
                
                if not user_input.strip():
                    continue

                # Используем execute_task для обработки ввода пользователя
                print(f"\033[93m{self.name} >\033[0m", end="", flush=True)
                final_response = self.execute_task(user_input)
                
                # Печатаем финальный ответ, который execute_task вернул
                # execute_task уже логирует промежуточные шаги
                print(final_response)

            except (KeyboardInterrupt, EOFError):
                print("\nЗавершение сеанса.")
                break


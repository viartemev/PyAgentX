"""
This module defines the Orchestrator, the central brain of the multi-agent team.
"""
import os
import json
import logging
from typing import Dict, Any
from openai import OpenAI
from app.factory.agent_factory import AgentFactory
from app.agents.roles.standard_roles import ALL_ROLES

ORCHESTRATOR_PROMPT_TEMPLATE = """
You are a master orchestrator of a team of AI agents.
Your job is to analyze the user's request and choose the best specialized agent to handle the task.

Here is the user's request:
"{user_query}"

And here is the team of specialists available to you:
{agents_description}

Based on the user's request, you must choose the single most appropriate agent to delegate the task to.
Respond with a JSON object containing the name of the chosen agent.

Example:
{{
  "agent_name": "FileSystemExpert"
}}
"""

class Orchestrator:
    """
    Analyzes user requests and routes them to the appropriate specialized agent.
    """
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.agent_factory = AgentFactory()
        self.api_key = api_key

    def _get_agents_description(self) -> str:
        """Creates a formatted string describing the available agents."""
        descriptions = []
        for name, config in ALL_ROLES.items():
            descriptions.append(f"- Agent: {name}\n  - Role: {config['role']}\n  - Best for: {config['goal']}")
        return "\n".join(descriptions)

    def _choose_agent(self, user_query: str) -> str:
        """
        Uses an LLM to choose the best agent for a given query.
        """
        agents_description = self._get_agents_description()
        prompt = ORCHESTRATOR_PROMPT_TEMPLATE.format(
            user_query=user_query,
            agents_description=agents_description
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            choice = json.loads(response.choices[0].message.content or "{}")
            agent_name = choice.get("agent_name")
            
            if agent_name in ALL_ROLES:
                logging.info(f"Orchestrator chose agent: {agent_name}")
                return agent_name
            else:
                logging.warning(f"Orchestrator chose an unknown agent: {agent_name}. Defaulting.")
                # Default to a generalist or the first available agent as a fallback
                return next(iter(ALL_ROLES))

        except Exception as e:
            logging.error(f"Error in choosing agent: {e}", exc_info=True)
            return next(iter(ALL_ROLES)) # Fallback to default

    def run(self, user_query: str) -> str:
        """
        Runs the orchestration process: choose an agent and execute the task.
        """
        # 1. Choose the right agent for the job
        chosen_agent_name = self._choose_agent(user_query)
        agent_config = ALL_ROLES[chosen_agent_name]

        # 2. Create the agent using the factory
        specialist_agent = self.agent_factory.create_agent(
            agent_config=agent_config,
            api_key=self.api_key,
            model=self.model
        )

        # 3. Execute the task with the chosen agent
        logging.info(f"Delegating task to {specialist_agent.name}...")
        final_response = specialist_agent.execute_task(user_query)
        
        return final_response
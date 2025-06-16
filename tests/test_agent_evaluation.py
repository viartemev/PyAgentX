"""
This module contains evaluation tests for the AI agent using the deepeval library.
"""
import os
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from dotenv import load_dotenv

from app.agents.agent import Agent

# Load environment variables from .env file
load_dotenv()

# Ensure necessary API keys are available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment or .env file.")

# Define a helpfulness metric for evaluation
helpfulness_metric = GEval(
    name="Helpfulness",
    criteria="Evaluate the helpfulness of the response. The response should be concise, relevant, and directly address the user's query.",
    evaluation_steps=[
        "Is the response relevant to the input?",
        "Does the response directly answer the core question?",
        "Is the response easy to understand and free of unnecessary jargon?",
    ],
    # It's recommended to use a powerful model for evaluation, like GPT-4
    # For this example, we use the same as the agent for simplicity.
    model="gpt-4o-mini",
)

@pytest.mark.skipif(not OPENAI_API_KEY, reason="OPENAI_API_KEY is not set")
def test_agent_web_search_and_summarize():
    """
    Tests the agent's ability to perform a web search and summarize the result.
    """
    # 1. Setup
    input_query = "Using the web search tool, find out what pytest is and provide a brief summary."
    
    # Initialize the agent
    agent = Agent(
        name="TestAgent",
        role="A testing assistant",
        goal="To successfully complete tasks for evaluation.",
        api_key=OPENAI_API_KEY,
        model="gpt-4o-mini"
    )

    # 2. Execution
    # We run the agent to get the actual output
    actual_output = agent.execute_task(input_query)
    
    # 3. Evaluation
    # We create a test case with the input and the agent's actual output
    test_case = LLMTestCase(
        input=input_query,
        actual_output=actual_output
    )
    
    # Assert that the output meets the helpfulness criteria
    # This will run the evaluation against the helpfulness_metric
    assert_test(test_case, [helpfulness_metric]) 
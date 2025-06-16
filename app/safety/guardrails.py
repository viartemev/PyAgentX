"""
This module defines the GuardrailManager for ensuring agent responses
adhere to predefined safety and topic constraints.
"""
import logging
from typing import Optional
from guardrails import Guard
from guardrails.hub import RestrictToTopic

class GuardrailManager:
    """
    Manages the validation of agent outputs using Guardrails AI.
    """
    def __init__(self):
        """Initializes the GuardrailManager with a predefined guard."""
        try:
            # Configure a guardrail to restrict conversation topics.
            # This is a simple example. More complex guardrails for toxicity,
            # PII, etc., can be added here.
            self.guard = Guard().use(
                RestrictToTopic(
                    valid_topics=["technology", "programming", "AI", "science"],
                    invalid_topics=["finance", "politics", "medical advice"],
                    disable_classifier=True, # Uses LLM-based classification
                    disable_llm=False,
                    on_fail="filter"
                )
            )
            logging.info("GuardrailManager initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize GuardrailManager: {e}", exc_info=True)
            self.guard = None

    def validate_response(self, response_text: str) -> Optional[str]:
        """
        Validates the agent's final response against the configured guardrails.

        Args:
            response_text: The final text response from the agent.

        Returns:
            The original response if validation passes, or a predefined message
            if it fails. Returns the original text if guardrails are not active.
        """
        if not self.guard:
            logging.warning("Guardrails are not active. Skipping validation.")
            return response_text

        try:
            validation_result = self.guard.validate(response_text)
            
            if validation_result.validation_passed:
                logging.info("Guardrail validation passed.")
                return response_text
            else:
                logging.warning(f"Guardrail validation failed: {validation_result.validation_summaries[0]}")
                return "I'm sorry, but I cannot discuss that topic. Please ask me about something else."

        except Exception as e:
            logging.error(f"An error occurred during guardrail validation: {e}", exc_info=True)
            return "I'm sorry, an error occurred while processing my response. Please try again." 
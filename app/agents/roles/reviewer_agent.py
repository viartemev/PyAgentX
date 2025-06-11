"""Reviewer Agent."""
from app.agents.agent import Agent
from app.rag.retriever import KnowledgeRetriever

class ReviewerAgent(Agent):
    """An agent specializing in strict Code Review."""
    def __init__(self, **kwargs):
        super().__init__(
            role="Code Reviewer",
            goal=(
                "Ensure that the provided code is of high quality, "
                "free of errors, and adheres to best practices and "
                "internal coding standards."
            ),
            **kwargs,
        )
        self.retriever = KnowledgeRetriever()

    def _get_system_prompt(self, **kwargs) -> str:
        code_to_review = kwargs.get("code", "")

        # Retrieve relevant knowledge
        retrieved_knowledge = self.retriever.retrieve(query=code_to_review, top_k=3)
        
        knowledge_context = "No specific internal standards found for this code."
        if retrieved_knowledge:
            formatted_knowledge = "\n\n---\n\n".join(
                [f"Source: {chunk['source']}\n\n{chunk['text']}" for chunk in retrieved_knowledge]
            )
            knowledge_context = (
                "When performing the review, pay close attention to the following "
                "internal standards and best practices:\n\n"
                f"--- RELEVANT KNOWLEDGE ---\n{formatted_knowledge}\n--------------------------"
            )

        return (
            f"You are a Senior Software Engineer acting as a code reviewer. "
            f"Your task is to provide a thorough review of the given code snippet.\n\n"
            f"{knowledge_context}\n\n"
            f"Please review the following code:\n\n"
            f"```python\n{code_to_review}\n```\n\n"
            f"Provide your feedback in a clear, constructive manner. "
            f"If you find issues, suggest specific improvements."
        )
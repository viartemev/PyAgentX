# tests/agents/test_reviewer_agent.py

import pytest
from unittest.mock import MagicMock, patch

from app.agents.roles.reviewer_agent import ReviewerAgent

@pytest.fixture
def mock_retriever_fixture():
    """Fixture to create a mock KnowledgeRetriever."""
    # Создаем мок для всего класса KnowledgeRetriever
    with patch('app.agents.roles.reviewer_agent.KnowledgeRetriever') as mock:
        # Настраиваем мок-экземпляр, который будет возвращаться при создании объекта
        mock_instance = MagicMock()
        # Настраиваем метод retrieve, чтобы он возвращал предсказуемые данные
        mock_instance.retrieve.return_value = [
            {
                "text": "Всегда используйте snake_case для переменных.",
                "source": "python_style_guide.md",
                "score": 0.95
            }
        ]
        # При создании KnowledgeRetriever() вернется наш настроенный мок
        mock.return_value = mock_instance
        yield mock_instance

def test_reviewer_agent_uses_rag_context(mock_retriever_fixture):
    """
    Tests that the ReviewerAgent correctly uses the context from the KnowledgeRetriever.
    """
    # Arrange
    # Инициализируем агента. Благодаря нашему фикстуре, self.retriever будет моком.
    agent = ReviewerAgent(name="TestReviewer", api_key="test_key")
    code_to_review = "my_variable = 1"
    
    # Act
    # Получаем системный промпт, который должен быть обогащен контекстом
    system_prompt = agent._get_system_prompt(code=code_to_review)

    # Assert
    # 1. Проверяем, что метод retrieve был вызван с правильным кодом
    mock_retriever_fixture.retrieve.assert_called_once_with(query=code_to_review, top_k=3)
    
    # 2. Проверяем, что информация из мока попала в системный промпт
    assert "RELEVANT KNOWLEDGE" in system_prompt
    assert "Всегда используйте snake_case для переменных." in system_prompt
    assert "python_style_guide.md" in system_prompt
    
    # 3. Проверяем, что если ретривер ничего не нашел, используется дефолтный текст
    mock_retriever_fixture.retrieve.return_value = []
    system_prompt_no_knowledge = agent._get_system_prompt(code=code_to_review)
    assert "No specific internal standards found" in system_prompt_no_knowledge 
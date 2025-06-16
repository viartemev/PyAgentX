# tests/agents/test_reviewer_agent.py

import pytest
from unittest.mock import MagicMock, patch
import json

from app.agents.roles.reviewer_agent import ReviewerAgent
from app.rag.retriever import KnowledgeRetriever

@pytest.fixture
def mock_retriever_fixture():
    """Fixture to create a mock KnowledgeRetriever."""
    # Создаем мок для всего класса KnowledgeRetriever
    with patch('app.agents.agent.KnowledgeRetriever') as mock:
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

def test_reviewer_agent_uses_rag_context(mock_retriever_fixture, mocker):
    """
    Tests that the ReviewerAgent correctly uses the context from the KnowledgeRetriever.
    """
    # Arrange
    agent = ReviewerAgent(
        name="TestReviewer",
        role="Test Reviewer", 
        goal="Review code",
        use_rag=True,
        api_key="fake_api_key",
    )
    code_to_review = "my_variable = 1"
    
    # Mock the model's response to stop the execution loop after one turn
    # Создаем корректную структуру ответа, которую ожидает json.loads
    mock_response_content = json.dumps({
        "thought": "The user wants me to review code. I will provide a final answer.",
        "answer": "The code `my_variable = 1` looks simple and correct."
    })
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = mock_response_content
    mocker.patch.object(agent.client.chat.completions, 'create', return_value=mock_completion)

    # Act
    # Запускаем execute_task, который теперь содержит логику обогащения промпта
    agent.execute_task(briefing=code_to_review)

    # Assert
    # 1. Проверяем, что метод retrieve был вызван с правильным запросом
    # _create_rag_query вернет сам брифинг, если не найдет шаблон задачи
    mock_retriever_fixture.retrieve.assert_called_once_with(query=code_to_review, top_k=3, filters=None)
    
    # 2. Проверяем, что информация из мока попала в системный промпт в истории диалога
    assert len(agent.conversation_history) > 0
    system_prompt_from_history = agent.conversation_history[0]['content']
    assert "RELEVANT KNOWLEDGE" in system_prompt_from_history
    assert "Всегда используйте snake_case для переменных." in system_prompt_from_history
    assert "python_style_guide.md" in system_prompt_from_history
    
    # 3. Проверяем, что если ретривер ничего не нашел, секция RELEVANT KNOWLEDGE отсутствует
    mock_retriever_fixture.retrieve.return_value = []
    mock_retriever_fixture.retrieve.reset_mock() # Сбрасываем мок
    
    agent_no_knowledge = ReviewerAgent(
        name="TestReviewer2",
        role="Test Reviewer",
        goal="Review code",
        use_rag=True,
        api_key="fake_api_key",
    )
    # Применяем тот же самый мок и ко второму агенту
    mocker.patch.object(agent_no_knowledge.client.chat.completions, 'create', return_value=mock_completion)

    agent_no_knowledge.execute_task(briefing=code_to_review)
    system_prompt_no_knowledge = agent_no_knowledge.conversation_history[0]['content']
    assert "RELEVANT KNOWLEDGE" not in system_prompt_no_knowledge
    assert "You are a Senior Software Engineer" in system_prompt_no_knowledge 
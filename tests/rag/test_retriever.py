# tests/rag/test_retriever.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from app.rag.retriever import KnowledgeRetriever

# Убираем фикстуру, так как будем создавать ретривер вручную
# def retriever() -> KnowledgeRetriever: ...

@pytest.fixture
def mock_retriever(mocker) -> KnowledgeRetriever:
    """
    Creates a completely mocked KnowledgeRetriever that does not depend on
    any external files.
    """
    # 1. Предотвращаем запуск реального __init__
    mocker.patch('app.rag.retriever.KnowledgeRetriever.__init__', return_value=None)
    
    # 2. Создаем "пустой" экземпляр класса
    retriever = KnowledgeRetriever()
    
    # 3. Вручную заполняем его данными, как будто они были загружены из файлов
    retriever.chunks = [
        {"source": "testing_guidelines.md", "text": "use pytest"},
        {"source": "error_handling.md", "text": "use try-except"},
        {"source": "python_style_guide.md", "text": "use snake_case"},
        {"source": "api_design.md", "text": "use REST"},
    ]
    retriever.embeddings = np.random.rand(4, 1536) # 4 чанка, 1536 измерений
    retriever.client = MagicMock() # Нам не нужен клиент, так как мы мокаем cosine_similarity
    
    return retriever

@pytest.mark.parametrize(
    "query, expected_source",
    [
        ("how to test?", "testing_guidelines.md"),
        ("how to handle errors?", "error_handling.md"),
        ("what case to use?", "python_style_guide.md"),
        ("how to design an api?", "api_design.md"),
    ],
)
def test_retriever_ranking_logic(
    mock_retriever: KnowledgeRetriever, query: str, expected_source: str, mocker
):
    """
    Tests that the retriever correctly sorts results based on similarity scores.
    """
    # Arrange
    num_chunks = len(mock_retriever.chunks)
    mock_similarities = np.linspace(0, 1, num_chunks) # [0.0, 0.25, 0.5, 1.0]
    np.random.shuffle(mock_similarities) # Перемешиваем

    # Ищем индекс нужного чанка и ставим ему максимальную схожесть
    target_index = [i for i, c in enumerate(mock_retriever.chunks) if c["source"] == expected_source][0]
    mock_similarities[target_index] = 1.1 # Гарантированно максимальное значение

    mocker.patch(
        'app.rag.retriever.cosine_similarity',
        return_value=np.array([mock_similarities])
    )
    # Так как `retrieve` вызывает `_get_embedding`, который вызывает `client`,
    # нам достаточно замокать вызов клиента.
    mock_retriever.client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=np.random.rand(1536).tolist())]
    )
    
    # Act
    retrieved_chunks = mock_retriever.retrieve(query, top_k=1)

    # Assert
    assert len(retrieved_chunks) == 1
    assert retrieved_chunks[0]["source"] == expected_source 
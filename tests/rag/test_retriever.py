# tests/rag/test_retriever.py
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from rank_bm25 import BM25Okapi

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
    retriever.openai_client = MagicMock()

    # 4. Инициализируем BM25, как это делается в методе load()
    tokenized_corpus = [chunk["text"].split(" ") for chunk in retriever.chunks]
    retriever.bm25 = BM25Okapi(tokenized_corpus)
    
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
    
    # Ищем индекс нужного чанка и ставим ему максимальную схожесть
    target_index = [i for i, c in enumerate(mock_retriever.chunks) if c["source"] == expected_source][0]
    
    # Мокаем _get_embedding, чтобы он возвращал предопределенный вектор для запроса
    # и разные векторы для чанков, где у целевого чанка будет наибольшая схожесть
    mock_query_embedding = np.random.rand(1536)
    
    # Создаем моки для эмбеддингов чанков
    mock_chunk_embeddings = np.random.rand(num_chunks, 1536)
    # Устанавливаем эмбеддинг целевого чанка так, чтобы он был идентичен запросу
    mock_chunk_embeddings[target_index] = mock_query_embedding
    mock_retriever.embeddings = mock_chunk_embeddings

    mocker.patch('app.rag.retriever.KnowledgeRetriever._get_embedding', return_value=mock_query_embedding)

    # Мокаем BM25 так, чтобы он не возвращал результатов и не влиял на фьюжн
    mock_retriever.bm25_index = MagicMock()
    mocker.patch.object(mock_retriever.bm25_index, 'get_scores', return_value=np.zeros(num_chunks))
    
    # Мокаем CrossEncoder, чтобы он не пытался скачаться и просто возвращал "как есть"
    mock_retriever.cross_encoder = None
    
    # Act
    retrieved_chunks = mock_retriever.retrieve(query, top_k=1)

    # Assert
    assert len(retrieved_chunks) == 1
    assert retrieved_chunks[0]["source"] == expected_source 
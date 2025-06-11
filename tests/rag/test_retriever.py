# tests/rag/test_retriever.py
import pytest
from app.rag.retriever import KnowledgeRetriever

@pytest.fixture(scope="module")
def retriever() -> KnowledgeRetriever:
    """
    Fixture to initialize the KnowledgeRetriever once for all tests in this module.
    This is efficient as the model and data are loaded only once.
    """
    try:
        return KnowledgeRetriever()
    except FileNotFoundError:
        pytest.fail(
            "Knowledge base not found. Please run "
            "'scripts/build_knowledge_base.py' first."
        )

@pytest.mark.parametrize(
    "query, expected_source",
    [
        (
            "как правильно называть тесты?",
            "testing_guidelines.md"
        ),
        (
            "как обрабатывать кастомные ошибки в коде?",
            "error_handling.md"
        ),
        (
            "когда использовать list comprehensions?",
            "python_style_guide.md"
        ),
        (
            "какие http методы использовать для создания ресурсов?",
            "api_design.md"
        )
    ],
)
def test_retriever_finds_correct_source(
    retriever: KnowledgeRetriever, query: str, expected_source: str
):
    """
    Tests that the retriever finds the most relevant document from the correct source.
    """
    # Act
    retrieved_chunks = retriever.retrieve(query, top_k=1)

    # Assert
    assert retrieved_chunks, f"Retriever found no chunks for query: '{query}'"
    assert len(retrieved_chunks) == 1
    
    top_chunk = retrieved_chunks[0]
    assert top_chunk["source"] == expected_source
    print(f"\nQuery: '{query}'\nFound in: '{top_chunk['source']}' with score {top_chunk['score']:.4f}") 
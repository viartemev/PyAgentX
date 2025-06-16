# tests/test_evaluation.py
import pytest
from unittest.mock import MagicMock, patch
import json

from app.evaluation.evaluator import CustomEvaluator

@pytest.fixture
def evaluator(mocker):
    """Fixture to create a CustomEvaluator with a mocked OpenAI client."""
    mocker.patch('os.getenv', return_value="fake_api_key")
    evaluator_instance = CustomEvaluator()
    # Mock the client within the instance
    evaluator_instance.client = MagicMock()
    return evaluator_instance

def test_evaluate_success(evaluator):
    """Tests a successful evaluation call."""
    # Arrange
    mock_response_content = json.dumps({
        "score": 5,
        "justification": "The answer is perfect."
    })
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = mock_response_content
    evaluator.client.chat.completions.create.return_value = mock_completion

    # Act
    result = evaluator.evaluate(
        actual_answer="The capital of France is Paris.",
        reference_answer="Paris is the capital of France.",
        criterion="Correctness"
    )

    # Assert
    assert result["score"] == 5
    assert result["justification"] == "The answer is perfect."
    evaluator.client.chat.completions.create.assert_called_once()

def test_evaluate_json_decode_error(evaluator):
    """Tests handling of a JSON decoding error from the LLM."""
    # Arrange
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "This is not valid JSON."
    evaluator.client.chat.completions.create.return_value = mock_completion
    
    # Act
    result = evaluator.evaluate("a", "b", "c")

    # Assert
    assert result["score"] == 0
    assert "Failed to parse" in result["justification"]

def test_evaluate_missing_keys(evaluator):
    """Tests handling of a response with missing keys."""
    # Arrange
    mock_response_content = json.dumps({"score": 4}) # Missing 'justification'
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = mock_response_content
    evaluator.client.chat.completions.create.return_value = mock_completion

    # Act
    result = evaluator.evaluate("a", "b", "c")

    # Assert
    assert result["score"] == 0
    assert "missing 'score' or 'justification'" in result["justification"]

def test_evaluate_empty_inputs():
    """Tests that the evaluator raises an error on empty inputs."""
    # No need to mock the client here, as it should fail before the API call.
    evaluator_instance = CustomEvaluator(api_key="fake_key")
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator_instance.evaluate(actual_answer="", reference_answer="b", criterion="c")

    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator_instance.evaluate(actual_answer="a", reference_answer="", criterion="c")
    
    with pytest.raises(ValueError, match="cannot be empty"):
        evaluator_instance.evaluate(actual_answer="a", reference_answer="b", criterion="") 
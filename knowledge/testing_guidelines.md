# Testing Guidelines

## 1. Test Structure: Arrange-Act-Assert (AAA)

All tests should follow the AAA pattern for clarity and readability.

- **Arrange:** Prepare all necessary data and mocks.
- **Act:** Call the function or method being tested.
- **Assert:** Check that the result meets expectations.

```python
def test_user_creation():
    # Arrange
    user_data = {"username": "test", "email": "test@example.com"}
    mock_db = MagicMock()

    # Act
    created_user = create_user(db=mock_db, data=user_data)

    # Assert
    assert created_user.username == user_data["username"]
    mock_db.add.assert_called_once()
```

## 2. Test Naming

Test function names should be descriptive and start with `test_`. Follow the format `test_<what_is_tested>_<under_what_conditions>_<expected_result>`.

**Example:** `test_add_items_with_negative_quantity_raises_error()`

## 3. Use `pytest.raises` for Exception Testing

To verify that code correctly raises exceptions, use the `pytest.raises` context manager.

```python
import pytest

def test_divide_by_zero_raises_exception():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)
``` 
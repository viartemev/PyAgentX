# Руководство по Написанию Тестов

## 1. Структура теста: Arrange-Act-Assert (AAA)

Все тесты должны следовать паттерну AAA для ясности и читаемости.

- **Arrange (Подготовка):** Подготовьте все необходимые данные и моки.
- **Act (Действие):** Вызовите тестируемую функцию или метод.
- **Assert (Проверка):** Проверьте, что результат соответствует ожиданиям.

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

## 2. Именование тестов

Имена тестовых функций должны быть описательными и начинаться с `test_`. Следуйте формату `test_<что_тестируем>_<при_каких_условиях>_<ожидаемый_результат>`.

**Пример:** `test_add_items_with_negative_quantity_raises_error()`

## 3. Используйте `pytest.raises` для проверки исключений

Для проверки того, что код корректно выбрасывает исключения, используйте контекстный менеджер `pytest.raises`.

```python
import pytest

def test_divide_by_zero_raises_exception():
    with pytest.raises(ZeroDivisionError):
        divide(10, 0)
``` 
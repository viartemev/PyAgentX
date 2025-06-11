# Error Handling Principles

## 1. Prefer Specific Exceptions

Always catch the most specific exception type possible. Avoid using `except Exception:` unless absolutely necessary.

**Bad:**
```python
try:
    # some code
except Exception as e:
    log.error("An error occurred")
```

**Good:**
```python
try:
    # some code
except FileNotFoundError as e:
    log.error(f"File not found: {e}")
except (KeyError, ValueError) as e:
    log.warning(f"Data error: {e}")
```

## 2. Use Custom Exceptions

For errors specific to your application's domain logic, create your own exception classes. This makes the code more readable and allows calling code to handle specific failures precisely.

```python
class InsufficientBalanceError(Exception):
    """Exception raised when the account balance is too low."""
    pass

def withdraw(amount):
    if amount > current_balance:
        raise InsufficientBalanceError("Insufficient funds in the account")
```

## 3. Log Errors Correctly

When catching an exception, be sure to log the full information, including the stack trace, to simplify debugging.

```python
import logging

try:
    # ...
except Exception as e:
    logging.error("An unexpected error occurred", exc_info=True) 
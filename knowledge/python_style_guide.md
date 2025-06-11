# Python Style Guide

- **Naming:** Use `snake_case` for variables and functions. Class names should use `CamelCase`. Constants should be in `UPPER_SNAKE_CASE`.
- **Line Length:** The maximum line length is 99 characters.
- **Docstrings:** All public modules, functions, classes, and methods must have Google-style docstrings.
- **Imports:** Group imports in the following order: standard library, third-party libraries, local application.

## String Formatting

- **f-strings:** Always prefer f-strings for formatting instead of `str.format()` or the `%` operator.

**Good:**
`user_info = f"User {user.name} with ID {user.id}"`

**Bad:**
`user_info = "User {} with ID {}".format(user.name, user.id)`

## List Comprehensions

- **Simplicity:** Use list comprehensions to create lists from existing iterables, but only if the logic remains simple and readable. If complex logic or multiple nested loops are required, use a regular `for` loop.

**Good:**
`squares = [x*x for x in range(10)]`

**Avoid (hard to read):**
`complex_list = [x + y for x in range(10) for y in range(5) if x % 2 == 0 if y % 3 == 0]` 
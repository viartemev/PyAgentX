# API Design Principles

- **Resources:** Use plural nouns for endpoint naming (e.g., `/users`, `/products`).
- **HTTP Methods:** Use the correct HTTP verbs for actions:
  - `GET` for retrieving data.
  - `POST` for creating new resources.
  - `PUT` / `PATCH` for updating.
  - `DELETE` for deleting.
- **Versioning:** Include the API version in the URL (e.g., `/api/v1/users`). 
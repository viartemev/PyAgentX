from fastapi import FastAPI
from typing import Dict

app: FastAPI = FastAPI(title="AI Agent API", version="0.1.0")

@app.get("/ping", response_model=Dict[str, str])
async def ping() -> Dict[str, str]:
    """Проверка работоспособности сервера.

    Returns:
        dict: Сообщение о статусе сервера.
    """
    return {"message": "pong"} 
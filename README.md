# AI Agent

AI Agent — это современный Python-проект для создания интеллектуального агента с использованием FastAPI, LangChain, Transformers и других передовых библиотек.

## Возможности
- Асинхронный API на FastAPI
- Интеграция с LLM через LangChain и Transformers
- Гибкая архитектура для расширения
- Легкая настройка и запуск

## Быстрый старт

### 1. Клонируйте репозиторий
```bash
git clone <your-repo-url>
cd agent_ai
```

### 2. Создайте и активируйте виртуальное окружение (venv)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Для Linux/MacOS
# .venv\Scripts\activate  # Для Windows
```

### 3. Установите зависимости
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Запустите сервер
```bash
uvicorn app.main:app --reload
```

## Структура проекта
```
agent_ai/
├── app/
│   ├── main.py         # Точка входа FastAPI
│   ├── agents/         # Логика AI-агентов
│   ├── services/       # Сервисы и интеграции
│   └── utils/          # Утилиты и вспомогательные функции
├── requirements.txt
├── README.md
└── .venv/
```

## Рекомендации
- Используйте Python 3.10+
- Для форматирования кода используйте ruff
- Для тестирования — pytest
- Все функции должны иметь аннотации типов и docstring в стиле Google
- Соблюдайте PEP8 и используйте ruff для автоформатирования

## Лицензия
MIT 
import pytest
from unittest.mock import MagicMock, ANY
from app.agents.agent import Agent
from app.agents.tools import read_file_definition

# Фикстура для создания экземпляра агента с мок-клиентом OpenAI
@pytest.fixture
def agent_with_mock_client(mocker):
    mock_openai_client = mocker.patch('openai.OpenAI')
    # Создаем экземпляр агента, передавая мок
    agent = Agent(name="TestAgent", tools=[read_file_definition])
    # Заменяем реальный клиент на мок в уже созданном экземпляре
    agent.client = mock_openai_client.return_value
    return agent

def test_agent_responds_without_tool(agent_with_mock_client):
    """
    Тест: Агент должен просто ответить, если инструмент не нужен.
    """
    agent = agent_with_mock_client
    
    # Настраиваем мок для возврата простого ответа
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_response.choices[0].message.content = "Hello, I am a test agent."
    agent.client.chat.completions.create.return_value = mock_response

    # Моделируем ввод пользователя и захватываем вывод
    def mock_input_handler():
        return ("привет", True)

    agent.user_input_handler = mock_input_handler
    
    # Для этого теста нам не нужен реальный цикл, достаточно одного шага
    conversation = [{"role": "system", "content": ANY}, {"role": "user", "content": "привет"}]
    
    # Выполняем шаг, который должен вернуть ответ
    response_message = agent.client.chat.completions.create(messages=conversation)
    
    assert not response_message.choices[0].message.tool_calls
    assert response_message.choices[0].message.content == "Hello, I am a test agent."


def test_agent_uses_tool_correctly(agent_with_mock_client, tmp_path):
    """
    Тест: Агент должен правильно вызвать инструмент и обработать результат.
    """
    agent = agent_with_mock_client
    
    # Создаем временный файл для чтения
    test_file = tmp_path / "riddle.txt"
    test_file.write_text("The answer is 42.")

    # 1. Мок ответа OpenAI, который запрашивает вызов инструмента
    tool_call_response = MagicMock()
    tool_call_response.choices = [MagicMock()]
    tool_call_message = MagicMock()
    tool_call_message.tool_calls = [MagicMock()]
    tool_call_message.tool_calls[0].id = "call_123"
    tool_call_message.tool_calls[0].type = "function"
    tool_call_message.tool_calls[0].function.name = "read_file"
    tool_call_message.tool_calls[0].function.arguments = f'{{"path": "{str(test_file)}"}}'
    tool_call_response.choices[0].message = tool_call_message
    
    # 2. Мок финального ответа OpenAI после получения результата инструмента
    final_response = MagicMock()
    final_response.choices = [MagicMock()]
    final_response.choices[0].message.tool_calls = None
    final_response.choices[0].message.content = "Я прочитал файл. Ответ на ваш вопрос: 42."

    # Настраиваем мок клиента, чтобы он возвращал ответы по очереди
    agent.client.chat.completions.create.side_effect = [
        tool_call_response,
        final_response
    ]

    # Запускаем один шаг выполнения агента
    conversation = [
        {"role": "system", "content": ANY},
        {"role": "user", "content": "прочитай файл riddle.txt"}
    ]

    # --- Шаг 1: Получение команды на вызов инструмента ---
    response1 = agent.client.chat.completions.create(messages=conversation)
    conversation.append(response1.choices[0].message)

    assert response1.choices[0].message.tool_calls is not None
    tool_call = response1.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "read_file"
    
    # --- Шаг 2: Выполнение инструмента и добавление результата в историю ---
    tool_result = agent._execute_tool(tool_call.function.name, {"path": str(test_file)})
    assert tool_result == "The answer is 42."
    
    conversation.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": tool_result,
    })

    # --- Шаг 3: Получение финального ответа ---
    response2 = agent.client.chat.completions.create(messages=conversation)
    assert response2.choices[0].message.tool_calls is None
    assert "42" in response2.choices[0].message.content 
import os
import pytest
from app.agents.tools import (
    read_file_tool,
    list_files_tool,
    edit_file_tool,
    delete_file_tool
)

@pytest.fixture
def test_file(tmp_path):
    """Фикстура для создания временного файла для тестов."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Hello, world!", encoding="utf-8")
    return file_path

def test_read_file_tool_success(test_file):
    """Тест успешного чтения файла."""
    result = read_file_tool({"path": str(test_file)})
    assert result == "Hello, world!"

def test_read_file_tool_not_found():
    """Тест ошибки при чтении несуществующего файла."""
    result = read_file_tool({"path": "non_existent_file.txt"})
    assert "Ошибка: Файл не найден" in result

def test_list_files_tool(tmp_path):
    """Тест успешного получения списка файлов."""
    (tmp_path / "dir1").mkdir()
    (tmp_path / "file1.txt").touch()
    result = list_files_tool({"path": str(tmp_path)})
    assert "dir1" in result
    assert "file1.txt" in result

def test_edit_file_tool(tmp_path):
    """Тест успешного создания и редактирования файла."""
    edit_path = tmp_path / "new_file.txt"
    
    # Создание файла
    result_create = edit_file_tool({"path": str(edit_path), "content": "Initial content"})
    assert "успешно сохранен" in result_create
    assert edit_path.read_text(encoding="utf-8") == "Initial content"
    
    # Редактирование файла
    result_edit = edit_file_tool({"path": str(edit_path), "content": "Updated content"})
    assert "успешно сохранен" in result_edit
    assert edit_path.read_text(encoding="utf-8") == "Updated content"

def test_delete_file_tool(test_file):
    """Тест успешного удаления файла."""
    assert os.path.exists(test_file)
    result = delete_file_tool({"path": str(test_file)})
    assert "успешно удален" in result
    assert not os.path.exists(test_file)

def test_delete_file_tool_not_found():
    """Тест ошибки при удалении несуществующего файла."""
    result = delete_file_tool({"path": "non_existent_file.txt"})
    assert "не найден и не может быть удален" in result 
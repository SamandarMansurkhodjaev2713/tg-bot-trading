import io
import os
import shutil
from .exceptions import FixError


def read_text(path: str, encoding: str = "utf-8") -> str:
    """Прочитать файл как текст с обработкой ошибок."""
    try:
        with io.open(path, "r", encoding=encoding) as f:
            return f.read()
    except FileNotFoundError as e:
        raise FixError(f"Файл не найден: {path}") from e
    except OSError as e:
        raise FixError(f"Ошибка чтения файла {path}: {e}") from e


def write_text(path: str, content: str, encoding: str = "utf-8") -> None:
    """Записать текст в файл с атомарным сохранением и бэкапом."""
    try:
        backup = path + ".bak"
        if os.path.exists(path):
            shutil.copy2(path, backup)
        tmp_path = path + ".tmp"
        with io.open(tmp_path, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp_path, path)
    except OSError as e:
        raise FixError(f"Ошибка записи файла {path}: {e}") from e
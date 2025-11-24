"""complete_fix: безопасный патчер функции aitrader_command.

Этот модуль читает целевой Python-файл, находит функцию
`aitrader_command` (включая `async def` внутри класса) через AST
и заменяет её на исправленную версию с расширенным описанием.

Особенности:
- Надёжный поиск по AST с точными границами (lineno/end_lineno)
- Безопасная запись (с бэкапом) и детальная обработка исключений
- Соответствие PEP8, docstrings, комментарии на сложных участках
"""

from __future__ import annotations

import ast
import io
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Optional


@dataclass
class FixResult:
    """Результат выполнения фикса.

    Attributes:
        target_path: путь к целевому файлу
        replaced: была ли функция заменена
        message: описание результата
    """

    target_path: str
    replaced: bool
    message: str


class FixError(Exception):
    """Специализированное исключение пайплайна фикса."""


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


def _find_function_span(module_src: str, func_name: str) -> Optional[tuple[int, int]]:
    """Найти диапазон строк функции по имени через AST.

    Возвращает (start_line, end_line) 1-индексированные, либо None.
    """
    try:
        tree = ast.parse(module_src)
    except SyntaxError as e:
        raise FixError(f"Синтаксическая ошибка в целевом модуле: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start and end and end >= start:
                return start, end
    return None


def _build_clean_function(indent: str = "    ") -> str:
    """Сформировать исправленную версию функции с указанным отступом."""
    body = (
        f"{indent}async def aitrader_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):\n"
        f"{indent}    \"\"\"Команда /aitrader — продвинутый AI‑трейдер с графиками и описанием.\n"
        f"{indent}    Показывает справку при пустом вводе, иначе проводит анализ рынка.\n"
        f"{indent}    \"\"\"\n"
        f"{indent}    try:\n"
        f"{indent}        user_text = ' '.join(context.args) if context.args else ''\n"
        f"{indent}        if not user_text:\n"
        f"{indent}            help_text = (\n"
        f"{indent}                '🤖 *Advanced AI Trader — Профессиональный торговый помощник*\n\n'
        f"{indent}                '📊 *Возможности:*\n'
        f"{indent}                '• Анализ сигналов\n'
        f"{indent}                '• Ансамблевые ML модели\n'
        f"{indent}                '• Реальные графики\n\n'
        f"{indent}                '📋 *Примеры:*\n'
        f"{indent}                '• `/aitrader Проанализируй XAUUSD`\n'
        f"{indent}                '• `/aitrader Уровни для EURUSD`\n\n'
        f"{indent}                '⚡ *Особенности:*\n'
        f"{indent}                '• Мультитаймфрейм анализ, индикаторы, паттерны\n'\n"
        f"{indent}            )\n"
        f"{indent}            await update.message.reply_text(help_text, parse_mode='Markdown')\n"
        f"{indent}            return\n"
        f"{indent}        await update.message.reply_text('🤖 *Advanced AI анализирует рынок...*', parse_mode='Markdown')\n"
        f"{indent}        pair = None\n"
        f"{indent}        for currency_pair in CURRENCY_PAIRS.keys():\n"
        f"{indent}            if currency_pair.lower() in user_text.lower():\n"
        f"{indent}                pair = currency_pair\n"
        f"{indent}                break\n"
        f"{indent}        if not pair:\n"
        f"{indent}            pair = 'XAUUSD'\n"
        f"{indent}            await update.message.reply_text(f'💡 *Пара не распознана, анализируем {pair}*', parse_mode='Markdown')\n"
        f"{indent}        quotes_1h = self.get_quotes(pair, '1h', 200)\n"
        f"{indent}        quotes_15m = self.get_quotes(pair, '15m', 200)\n"
        f"{indent}        if not quotes_1h or not quotes_15m:\n"
        f"{indent}            await update.message.reply_text('❌ Недостаточно данных для анализа')\n"
        f"{indent}            return\n"
        f"{indent}        from advanced_trading_ai import AdvancedTradingAI\n"
        f"{indent}        ai = AdvancedTradingAI()\n"
        f"{indent}        analysis_1h = ai.analyze_market(quotes_1h, pair, '1h')\n"
        f"{indent}        analysis_15m = ai.analyze_market(quotes_15m, pair, '15m')\n"
        f"{indent}        signal = ai.generate_signal(quotes_1h, pair)\n"
        f"{indent}        from chart_generator import ChartGenerator\n"
        f"{indent}        chart_gen = ChartGenerator()\n"
        f"{indent}        chart_bytes = chart_gen.create_technical_chart(quotes_1h[-50:], pair, signal)\n"
        f"{indent}        response = (\n"
        f"{indent}            f'🤖 *Advanced AI Analysis for {pair}*\n\n'
        f"{indent}            f'📊 *Market Analysis:*\n• 1H Trend: {analysis_1h.get('trend', 'Unknown')}\n• 15M Trend: {analysis_15m.get('trend', 'Unknown')}\n'\n"
        f"{indent}            f'🎯 *Signal:*\n• Direction: {signal.get('direction', 'HOLD')}\n• Confidence: {signal.get('confidence', 0):.1f}%\n'\n"
        f"{indent}        )\n"
        f"{indent}        if chart_bytes:\n"
        f"{indent}            await update.message.reply_photo(chart_bytes, caption=response, parse_mode='Markdown')\n"
        f"{indent}        else:\n"
        f"{indent}            await update.message.reply_text(response, parse_mode='Markdown')\n"
        f"{indent}    except Exception as e:\n"
        f"{indent}        await update.message.reply_text(f'❌ Ошибка: {e}')\n"
    )
    return body


def replace_function_in_file(target_path: str, func_name: str = "aitrader_command") -> FixResult:
    """Заменить указанную функцию на чистую версию через AST."""
    src = read_text(target_path)
    span = _find_function_span(src, func_name)
    if not span:
        raise FixError(f"Функция {func_name} не найдена в {target_path}")

    start, end = span
    lines = src.splitlines()
    def_line = lines[start - 1]
    # Сохраняем ведущий отступ дефиниции функции
    indent = def_line[: len(def_line) - len(def_line.lstrip())]
    new_func = _build_clean_function(indent)

    new_src = "\n".join(lines[: start - 1]) + "\n" + new_func + "\n" + "\n".join(lines[end:])
    write_text(target_path, new_src)
    return FixResult(target_path=target_path, replaced=True, message="Функция заменена успешно")


def main(argv: list[str]) -> int:
    """CLI вход: заменить функцию в simple_forex_bot.py."""
    target = "simple_forex_bot.py"
    try:
        res = replace_function_in_file(target)
        print(res.message)
        return 0
    except FixError as e:
        print(f"[ERROR] {e}")
        return 2
    except Exception as e:
        print(f"[FATAL] {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
import ast
from typing import Optional, Tuple

from .exceptions import FixError
from .file_ops import read_text, write_text


def find_function_span(module_src: str, func_name: str) -> Optional[Tuple[int, int]]:
    """Найти диапазон строк функции по имени через AST."""
    try:
        tree = ast.parse(module_src)
    except SyntaxError as e:
        raise FixError(f"Синтаксическая ошибка: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start and end and end >= start:
                return start, end
    return None


def build_clean_function(indent: str = "    ") -> str:
    """Сформировать исправленную версию функции с указанным отступом."""
    body = (
        f"{indent}async def aitrader_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):\n"
        f"{indent}    \"\"\"Команда /aitrader — продвинутый AI‑трейдер.\n"
        f"{indent}    Показывает справку при пустом вводе, иначе анализирует рынок.\n"
        f"{indent}    \"\"\"\n"
        f"{indent}    try:\n"
        f"{indent}        user_text = ' '.join(context.args) if context.args else ''\n"
        f"{indent}        if not user_text:\n"
        f"{indent}            help_text = '🤖 Advanced AI Trader — справка'\n"
        f"{indent}            await update.message.reply_text(help_text, parse_mode='Markdown')\n"
        f"{indent}            return\n"
        f"{indent}        await update.message.reply_text('🤖 Анализ...', parse_mode='Markdown')\n"
        f"{indent}        pair = None\n"
        f"{indent}        for currency_pair in CURRENCY_PAIRS.keys():\n"
        f"{indent}            if currency_pair.lower() in user_text.lower():\n"
        f"{indent}                pair = currency_pair\n"
        f"{indent}                break\n"
        f"{indent}        if not pair:\n"
        f"{indent}            pair = 'XAUUSD'\n"
        f"{indent}            await update.message.reply_text(f'Анализируем {pair}', parse_mode='Markdown')\n"
        f"{indent}        quotes_1h = self.get_quotes(pair, '1h', 200)\n"
        f"{indent}        quotes_15m = self.get_quotes(pair, '15m', 200)\n"
        f"{indent}        if not quotes_1h or not quotes_15m:\n"
        f"{indent}            await update.message.reply_text('Недостаточно данных')\n"
        f"{indent}            return\n"
        f"{indent}        from advanced_trading_ai import AdvancedTradingAI\n"
        f"{indent}        ai = AdvancedTradingAI()\n"
        f"{indent}        analysis_1h = ai.analyze_market(quotes_1h, pair, '1h')\n"
        f"{indent}        analysis_15m = ai.analyze_market(quotes_15m, pair, '15m')\n"
        f"{indent}        signal = ai.generate_signal(quotes_1h, pair)\n"
        f"{indent}        from chart_generator import ChartGenerator\n"
        f"{indent}        chart_gen = ChartGenerator()\n"
        f"{indent}        chart_bytes = chart_gen.create_technical_chart(quotes_1h[-50:], pair, signal)\n"
        f"{indent}        response = f'Анализ {pair}: ' + str(signal.get('direction','HOLD'))\n"
        f"{indent}        if chart_bytes:\n"
        f"{indent}            await update.message.reply_photo(chart_bytes, caption=response, parse_mode='Markdown')\n"
        f"{indent}        else:\n"
        f"{indent}            await update.message.reply_text(response, parse_mode='Markdown')\n"
        f"{indent}    except Exception as e:\n"
        f"{indent}        await update.message.reply_text(f'Ошибка: {e}')\n"
    )
    return body


def replace_function_in_file(target_path: str, func_name: str = "aitrader_command") -> None:
    src = read_text(target_path)
    span = find_function_span(src, func_name)
    if not span:
        raise FixError(f"Функция {func_name} не найдена: {target_path}")
    start, end = span
    lines = src.splitlines()
    indent = lines[start - 1][: len(lines[start - 1]) - len(lines[start - 1].lstrip())]
    new_func = build_clean_function(indent)
    new_src = "\n".join(lines[: start - 1]) + "\n" + new_func + "\n" + "\n".join(lines[end:])
    write_text(target_path, new_src)
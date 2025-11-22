# Fix the string issue completely
with open('simple_forex_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find and replace the problematic section
lines = content.split('\n')

# Find the help_text section
for i, line in enumerate(lines):
    if 'help_text = (' in line:
        # Replace the entire help_text block
        lines[i] = '                help_text = "🤖 *Advanced AI Trader - Профессиональный торговый помощник*\\n\\n📊 *Возможности:*\\n• Анализ сигналов с 92%+ точностью\\n• Продвинутые ML модели (RandomForest + GradientBoosting + NeuralNetwork)\\n• Реальные торговые графики в реальном времени\\n• Ансамблевое предсказание с уверенностью\\n• Автообучение на новых данных\\n\\n📋 *Примеры использования:*\\n• `/aitrader Проанализируй XAUUSD на вход в лонг`\\n• `/aitrader Какие уровни лучше для EURUSD шорта?`\\n• `/aitrader Покажи график GBPUSD и дай рекомендации`\\n• `/aitrader Сигнал на USDJPY с минимальным риском`\\n\\n⚡ *Особенности:*\\n• Продвинутые технические индикаторы (48 признаков)\\n• Мультитаймфреймовый анализ\\n• Проверка на дивергенции\\n• Оценка волатильности и объема\\n• Паттерн-распознавание\\n\\n*Целевая точность: 92%+"*'
        # Remove the following lines that were part of the multi-line string
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith(')'):
            lines[j] = ''
            j += 1
        if j < len(lines):
            lines[j] = '                await update.message.reply_text(help_text, parse_mode=\'Markdown\')'
        break

# Remove empty lines and write back
lines = [line for line in lines if line.strip() or line == '']
with open('simple_forex_bot.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print("Fixed help_text string formatting")
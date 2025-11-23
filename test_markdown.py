#!/usr/bin/env python3
"""
Тестирование Markdown parsing в aitrader команде
"""

def test_markdown_response():
    """Тестирование Markdown разметки в ответе aitrader"""
    
    # Создаем тестовые данные аналогичные реальным
    pair = "BTC"
    current_price = 84288.32812
    ml_prediction = {'signal': 1, 'confidence': 0.85, 'probability': 0.78}
    signal_alignment = "совпадает"
    market_analysis_1h = {'trend': 'bullish'}
    market_analysis_15m = {'trend': 'bullish'}
    market_analysis_1d = {'trend': 'bearish'}
    trend_score = 3
    
    # Формируем ответ как в функции aitrader
    signal_text = 'ЛОНГ'
    confidence_emoji = '🟢'
    trend_1h = '🟢 Восходящий'
    trend_15m = '🟢 Восходящий'
    trend_1d = '🔴 Нисходящий'
    trend_emoji = '🟢'
    
    response = f"""🤖 *Advanced AI Trader Analysis - {pair}*

📊 *Текущая рыночная ситуация:*
• Текущая цена: {current_price:.5f}
• ML Сигнал: {signal_text}
• Уверенность AI: {ml_prediction['confidence']*100:.1f}% {confidence_emoji}
• Вероятность успеха: {ml_prediction['probability']*100:.1f}%
• Сигнал пользователя: {signal_alignment}

📈 *Мультитаймфреймовый анализ:*
• 1H Тренд: {trend_1h}
• 15M Тренд: {trend_15m}
• 1D Тренд: {trend_1d}
• Оценка тренда: {trend_score}/5 {trend_emoji}"""
    
    print("📊 Тестовый ответ:")
    print(response)
    print(f"\n📏 Длина ответа: {len(response)} символов")
    print(f"📏 Длина в байтах: {len(response.encode('utf-8'))} байт")
    
    # Проверяем символ на позиции 1213
    byte_offset = 1213
    response_bytes = response.encode('utf-8')
    
    if len(response_bytes) > byte_offset:
        print(f"\n🔍 Анализ байта на позиции {byte_offset}:")
        print(f"Код байта: {response_bytes[byte_offset]}")
        print(f"Символ: {chr(response_bytes[byte_offset])}")
        
        # Показываем контекст вокруг проблемного места
        start = max(0, byte_offset - 20)
        end = min(len(response_bytes), byte_offset + 20)
        context = response_bytes[start:end].decode('utf-8', errors='ignore')
        print(f"\n📝 Контекст вокруг позиции {byte_offset}:")
        print(f"'{context}'")
    
    # Проверяем на проблемные символы
    print(f"\n🔍 Проверка на проблемные символы:")
    problematic_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    for i, char in enumerate(response):
        if char in problematic_chars:
            print(f"Позиция {i}: '{char}' - может вызывать проблемы")
    
    return response

def test_telegram_markdown():
    """Тестирование Telegram Markdown ограничений"""
    
    print("\n🧪 Тестирование Telegram Markdown ограничений:")
    
    # Проверяем emoji символы
    emoji_test = "🟢 🔴 🟡 📈 📉 🤖 📊"
    print(f"Emoji тест: {emoji_test}")
    print(f"Длина в байтах: {len(emoji_test.encode('utf-8'))}")
    
    # Проверяем форматирование
    markdown_test = """*жирный текст*
_курсивный текст_
`код`
[ссылка](https://example.com)"""
    
    print(f"\nMarkdown тест:\n{markdown_test}")

if __name__ == "__main__":
    test_markdown_response()
    test_telegram_markdown()
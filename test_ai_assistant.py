#!/usr/bin/env python3
"""
Тестирование AI ассистента для торговли
"""

import asyncio
from datetime import datetime
from simple_forex_bot import TradingAIAssistant

async def test_ai_assistant():
    """Тестируем AI ассистента"""
    
    # Создаем экземпляр AI ассистента
    ai = TradingAIAssistant()
    
    # Тестовые сигналы
    test_signals = [
        "Хочу лонг XAUUSD со стопом 2650 и тейком 2720",
        "Думаю открыть шорт EURUSD, стоп 1.0850, тейк 1.0750",
        "Планирую лонг GBPUSD с риском 2%, стоп 1.2450, тейк 1.2550",
        "Хочу купить золото, цель 2700, стоп 2600"
    ]
    
    print("🤖 Тестирование AI ассистента для торговли\n")
    
    for signal in test_signals:
        print(f"📨 Входной сигнал: {signal}")
        
        # Парсим сигнал
        parsed = ai.parse_signal_request(signal)
        print(f"📊 Распознанные данные: {parsed}")
        
        if parsed['success']:
            # Создаем фиктивные данные для теста
            market_analysis = {
                'current_price': 2655.0,
                'rsi': 65.0,
                'macd_signal': 'bullish',
                'bb_position': 0.5,  # Значение от 0 до 1 (0 = нижняя полоса, 1 = верхняя полоса)
                'atr': 15.0,
                'trend': 'uptrend',
                'volatility': 'moderate',
                'support_levels': [2650, 2640, 2630],
                'resistance_levels': [2670, 2680, 2700]
            }
            
            pair_specs = {
                'spread': 0.2,
                'commission': 0.0,
                'swap_long': -2.5,
                'swap_short': 0.5,
                'leverage': 100
            }
            
            # Оцениваем сигнал
            evaluation = ai.evaluate_signal(parsed, market_analysis, pair_specs)
            
            print(f"🎯 Оценка сигнала:")
            print(f"   Счет: {evaluation['score']}/100")
            print(f"   Рекомендация: {evaluation['recommendation']}")
            print(f"   Риск-вознаграждение: {evaluation['risk_reward_ratio']}")
            print(f"   Обратная связь: {evaluation['feedback']}")
            
            if evaluation['warnings']:
                print(f"   ⚠️  Предупреждения: {', '.join(evaluation['warnings'])}")
            
            if evaluation['recommendations']:
                print(f"   💡 Рекомендации: {', '.join(evaluation['recommendations'])}")
        
        print("-" * 50)
        await asyncio.sleep(1)  # Небольшая пауза между тестами

if __name__ == "__main__":
    asyncio.run(test_ai_assistant())
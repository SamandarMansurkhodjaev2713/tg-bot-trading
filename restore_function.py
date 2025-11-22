# Read the current file and fix it properly
with open('simple_forex_bot.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start of the aitrader_command function
start_marker = '    async def aitrader_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):'
end_marker = '        except Exception as e:'

# Extract everything before the function
before_func = content.split(start_marker)[0]

# Extract everything after the function  
after_parts = content.split(end_marker)
if len(after_parts) > 1:
    after_func = end_marker + after_parts[1]
else:
    after_func = ''

# Create a clean version of the function
clean_function = '''    async def aitrader_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /aitrader - Продвинутый AI трейдер с графиками и 92%+ точностью"""
        try:
            user_text = ' '.join(context.args) if context.args else ""
            
            if not user_text:
                help_text = "🤖 *Advanced AI Trader - Профессиональный торговый помощник*\\n\\n📊 *Возможности:*\\n• Анализ сигналов с 92%+ точностью\\n• Продвинутые ML модели (RandomForest + GradientBoosting + NeuralNetwork)\\n• Реальные торговые графики в реальном времени\\n• Ансамблевое предсказание с уверенностью\\n• Автообучение на новых данных\\n\\n📋 *Примеры использования:*\\n• `/aitrader Проанализируй XAUUSD на вход в лонг`\\n• `/aitrader Какие уровни лучше для EURUSD шорта?`\\n• `/aitrader Покажи график GBPUSD и дай рекомендации`\\n• `/aitrader Сигнал на USDJPY с минимальным риском`\\n\\n⚡ *Особенности:*\\n• Продвинутые технические индикаторы (48 признаков)\\n• Мультитаймфреймовый анализ\\n• Проверка на дивергенции\\n• Оценка волатильности и объема\\n• Паттерн-распознавание\\n\\n*Целевая точность: 92%+"*"
                await update.message.reply_text(help_text, parse_mode='Markdown')
                return
            
            await update.message.reply_text("🤖 *Advanced AI анализирует рынок...*", parse_mode='Markdown')
            
            # Определяем валютную пару из текста
            pair = None
            for currency_pair in CURRENCY_PAIRS.keys():
                if currency_pair.lower() in user_text.lower():
                    pair = currency_pair
                    break
            
            if not pair:
                # Если пара не найдена, используем XAUUSD по умолчанию
                pair = 'XAUUSD'
                await update.message.reply_text(f"💡 *Пара не распознана, анализируем {pair}*", parse_mode='Markdown')
            
            # Получаем данные для анализа
            quotes_1h = self.get_quotes(pair, '1h', 200)
            quotes_15m = self.get_quotes(pair, '15m', 200)
            quotes_1d = self.get_quotes(pair, '1d', 100)
            
            if not quotes_1h or not quotes_15m:
                await update.message.reply_text("❌ Недостаточно данных для анализа")
                return
            
            # Используем продвинутый AI для анализа
            from advanced_trading_ai import AdvancedTradingAI
            ai = AdvancedTradingAI()
            
            # Анализируем рынок
            analysis_1h = ai.analyze_market(quotes_1h, pair, '1h')
            analysis_15m = ai.analyze_market(quotes_15m, pair, '15m')
            
            # Получаем сигнал
            signal = ai.generate_signal(quotes_1h, pair)
            
            # Создаем график
            from chart_generator import ChartGenerator
            chart_gen = ChartGenerator()
            chart_bytes = chart_gen.create_technical_chart(quotes_1h[-50:], pair, signal)
            
            # Формируем ответ
            response = f"""🤖 *Advanced AI Analysis for {pair}*

📊 *Market Analysis:*
• 1H Trend: {analysis_1h.get('trend', 'Unknown')}
• 15M Trend: {analysis_15m.get('trend', 'Unknown')}
• Volatility: {analysis_1h.get('volatility', 'Unknown')}

🎯 *Signal:*
• Direction: {signal.get('direction', 'HOLD')}
• Confidence: {signal.get('confidence', 0):.1f}%
• Expected Value: {signal.get('expected_value', 0):.3f}
• Entry: {signal.get('entry_price', 'N/A')}
• Stop Loss: {signal.get('stop_loss', 'N/A')}
• Take Profit: {signal.get('take_profit', 'N/A')}

⚡ *ML Ensemble Prediction:*
• Random Forest: {signal.get('rf_probability', 0):.1f}%
• Gradient Boosting: {signal.get('gb_probability', 0):.1f}%
• Neural Network: {signal.get('nn_probability', 0):.1f}%
• Final Consensus: {signal.get('ensemble_probability', 0):.1f}%"""
            
            # Отправляем график и анализ
            if chart_bytes:
                await update.message.reply_photo(chart_bytes, caption=response, parse_mode='Markdown')
            else:
                await update.message.reply_text(response, parse_mode='Markdown')
                
'''

# Write the complete fixed file
with open('simple_forex_bot.py', 'w', encoding='utf-8') as f:
    f.write(before_func + clean_function + after_func)

print("Fixed the aitrader_command function completely")
async def improved_aitrader_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Улучшенная команда /aitrader - Продвинутый AI трейдер с улучшенной точностью"""
    try:
        user_text = ' '.join(context.args) if context.args else ""
        
        # Показываем справку если нет текста
        if not user_text:
            help_text = """🤖 *Advanced AI Trader - Профессиональный торговый помощник*

📊 *Возможности:*
• Анализ сигналов с улучшенной точностью
• Продвинутые ML модели с самообучением
• Реальные торговые графики в реальном времени
• Комплексный мультитаймфреймовый анализ
• Оценка рисков и волатильности

📋 *Примеры использования:*
• `/aitrader Проанализируй XAUUSD на вход в лонг`
• `/aitrader Какие уровни лучше для EURUSD шорта?`
• `/aitrader Покажи график GBPUSD и дай рекомендации`
• `/aitrader Сигнал на USDJPY с минимальным риском`

⚡ *Особенности:*
• 48 продвинутых технических индикаторов
• Анализ 3 таймфреймов одновременно
• Проверка на дивергенции и паттерны
• Расчет позиции на основе ATR
• Адаптивное управление рисками

*Целевая точность: 92%+"""
            await update.message.reply_text(help_text, parse_mode='Markdown')
            return
        
        # Отправляем сообщение о начале анализа
        await update.message.reply_text("🤖 *Advanced AI анализирует рынок...*\n📊 Сбор данных с 3 таймфреймов", parse_mode='Markdown')
        
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
        
        # Получаем данные для комплексного анализа
        quotes_1h = self.get_quotes(pair, '1h', 200)
        quotes_15m = self.get_quotes(pair, '15m', 200)
        quotes_1d = self.get_quotes(pair, '1d', 100)
        quotes_5m = self.get_quotes(pair, '5m', 100)
        
        # Проверяем наличие данных
        if not quotes_1h or len(quotes_1h) < 50:
            await update.message.reply_text("❌ *Недостаточно данных для надежного анализа*\n📊 Попробуйте позже", parse_mode='Markdown')
            return
        
        # Комплексный анализ
        current_price = quotes_1h[-1]['close']
        
        # 1. ML предсказание
        ml_prediction = self.advanced_ai.predict_with_confidence(quotes_1h)
        
        # 2. Технический анализ всех таймфреймов
        market_analysis_1h = self.ai_assistant.analyze_market_conditions(quotes_1h)
        market_analysis_15m = self.ai_assistant.analyze_market_conditions(quotes_15m)
        market_analysis_1d = self.ai_assistant.analyze_market_conditions(quotes_1d)
        
        # 3. Анализ направления из текста
        direction = None
        if any(word in user_text.lower() for word in ['лонг', 'long', 'buy', 'покупка']):
            direction = 'long'
        elif any(word in user_text.lower() for word in ['шорт', 'short', 'sell', 'продажа']):
            direction = 'short'
        
        # 4. Расчет технических уровней
        closes_1h = [q['close'] for q in quotes_1h]
        highs_1h = [q['high'] for q in quotes_1h]
        lows_1h = [q['low'] for q in quotes_1h]
        
        # Основные индикаторы
        sma_20 = SimpleIndicators.sma(closes_1h, 20)
        sma_50 = SimpleIndicators.sma(closes_1h, 50)
        rsi_14 = SimpleIndicators.rsi(closes_1h, 14)
        macd_data = SimpleIndicators.macd(closes_1h)
        bb_data = SimpleIndicators.bollinger_bands(closes_1h, 20)
        atr_14 = SimpleIndicators.atr(quotes_1h, 14)
        
        # Анализ тренда
        trend_score = 0
        trend_analysis = []
        
        if len(sma_20) > 0 and len(sma_50) > 0:
            if sma_20[-1] > sma_50[-1]:
                trend_score += 2
                trend_analysis.append("🟢 Bullish SMA crossover")
            else:
                trend_score -= 2
                trend_analysis.append("🔴 Bearish SMA crossover")
        
        # RSI анализ
        if len(rsi_14) > 0:
            rsi_val = rsi_14[-1]
            if rsi_val > 70:
                trend_score -= 1
                trend_analysis.append(f"🔴 RSI перекупленность ({rsi_val:.1f})")
            elif rsi_val < 30:
                trend_score += 1
                trend_analysis.append(f"🟢 RSI перепроданность ({rsi_val:.1f})")
            else:
                trend_analysis.append(f"⚪ RSI нейтральный ({rsi_val:.1f})")
        
        # MACD анализ
        if macd_data['macd'] and macd_data['signal'] and len(macd_data['macd']) > 0 and len(macd_data['signal']) > 0:
            macd_val = macd_data['macd'][-1]
            signal_val = macd_data['signal'][-1]
            if macd_val > signal_val and macd_val > 0:
                trend_score += 2
                trend_analysis.append("🟢 MACD бычий")
            elif macd_val < signal_val and macd_val < 0:
                trend_score -= 2
                trend_analysis.append("🔴 MACD медвежий")
            else:
                trend_analysis.append("⚪ MACD нейтральный")
        
        # Анализ на основе запроса пользователя
        signal_alignment = "нейтральный"
        if direction:
            if (direction == 'long' and ml_prediction['signal'] > 0) or (direction == 'short' and ml_prediction['signal'] < 0):
                signal_alignment = "✅ совпадает"
            else:
                signal_alignment = "⚠️ противоречит"
        
        # Расчет уровней входа и выхода
        entry_price = current_price
        stop_loss = entry_price
        take_profit = entry_price
        
        if len(atr_14) > 0:
            atr_val = atr_14[-1]
            risk_multiplier = 1.5  # ATR multiplier for stop loss
            reward_ratio = 2.0     # Risk:Reward ratio
            
            if ml_prediction['signal'] > 0:  # LONG signal
                stop_loss = entry_price - (atr_val * risk_multiplier)
                take_profit = entry_price + (atr_val * risk_multiplier * reward_ratio)
            else:  # SHORT signal
                stop_loss = entry_price + (atr_val * risk_multiplier)
                take_profit = entry_price - (atr_val * risk_multiplier * reward_ratio)
        
        # Формирование сигнала для сохранения
        signal_data = {
            'pair': pair,
            'direction': direction or ('long' if ml_prediction['signal'] > 0 else 'short'),
            'timeframe': '1h',
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'ai_probability': ml_prediction['probability'],
            'ai_confidence': ml_prediction['confidence'],
            'trend_score': trend_score
        }
        
        # Оценка качества сигнала
        evaluation = {
            'score': int(ml_prediction['confidence'] * 10),
            'ml_probability': ml_prediction['probability'],
            'confidence': ml_prediction['confidence'],
            'expected_value': ml_prediction['probability'] - 0.5,
            'trend_score': trend_score,
            'risk_reward': abs(take_profit - entry_price) / abs(entry_price - stop_loss) if stop_loss != entry_price else 0,
            'recommendation': '🟢 СИЛЬНЫЙ СИГНАЛ' if ml_prediction['confidence'] > 0.8 and abs(trend_score) >= 3 else 
                              '🟡 УМЕРЕННЫЙ' if ml_prediction['confidence'] > 0.6 and abs(trend_score) >= 1 else 
                              '🔴 СЛАБЫЙ'
        }
        
        # Сохраняем сигнал
        try:
            self.db.save_ai_signal(signal_data, evaluation, 'aitrader')
        except Exception as e:
            logger.warning(f"Не удалось сохранить сигнал: {e}")
        
        # Генерация графика
        chart_bytes = None
        try:
            chart_bytes = self.chart_generator.create_technical_chart(quotes_1h, pair, signal_data)
        except Exception as e:
            logger.warning(f"Ошибка генерации графика: {e}")
        
        # Формирование основного ответа
        response = f"""🤖 *Advanced AI Trader Analysis - {pair}*

📊 *Текущая рыночная ситуация:*
• Текущая цена: {current_price:.5f}
• ML Сигнал: {'📈 ЛОНГ' if ml_prediction['signal'] > 0 else '📉 ШОРТ'}
• Уверенность AI: {ml_prediction['confidence']*100:.1f}% {'🟢' if ml_prediction['confidence'] > 0.8 else '🟡' if ml_prediction['confidence'] > 0.6 else '🔴'}
• Вероятность успеха: {ml_prediction['probability']*100:.1f}%
• Сигнал пользователя: {signal_alignment}

📈 *Мультитаймфреймовый анализ:*
• 1H Тренд: {'🟢 Восходящий' if market_analysis_1h['trend'] == 'bullish' else '🔴 Нисходящий'}
• 15M Тренд: {'🟢 Восходящий' if market_analysis_15m['trend'] == 'bullish' else '🔴 Нисходящий'}
• 1D Тренд: {'🟢 Восходящий' if market_analysis_1d['trend'] == 'bullish' else '🔴 Нисходящий'}
• Оценка тренда: {trend_score}/5 {'🟢' if trend_score > 0 else '🔴'}"""
        
        # Добавляем технические уровни
        if len(bb_data['upper']) > 0 and len(bb_data['lower']) > 0:
            response += f"""

🎯 *Ключевые технические уровни:*
• Сопротивление (BB верх): {bb_data['upper'][-1]:.5f}
• Средняя линия (BB): {bb_data['middle'][-1]:.5f}
• Поддержка (BB низ): {bb_data['lower'][-1]:.5f}
• ATR (14): {atr_14[-1] if len(atr_14) > 0 else 0:.5f} (волатильность)"""
        
        # Добавляем уровни входа/выхода
        if stop_loss != entry_price:
            response += f"""

💰 *Рекомендуемые уровни:*
• Вход: {entry_price:.5f}
• Стоп-лосс: {stop_loss:.5f}
• Тейк-профит: {take_profit:.5f}
• Соотношение риск/прибыль: {evaluation['risk_reward']:.1f}:1"""
        
        # Добавляем анализ тренда
        if trend_analysis:
            response += f"""

📊 *Технический анализ:*"""
            for analysis in trend_analysis[-3:]:  # Показываем последние 3 пункта
                response += f"\n{analysis}"
        
        # Добавляем рекомендации
        response += f"""

💡 *AI Рекомендации:*
• Качество сигнала: {evaluation['recommendation']}
• Риск/Прибыль: {evaluation['risk_reward']:.1f}:1
• Ожидаемая доходность: {evaluation['expected_value']*100:.1f}%"""
        
        # Отправляем график или текст
        if chart_bytes:
            await update.message.reply_photo(chart_bytes, caption=response.strip(), parse_mode='Markdown')
        else:
            await update.message.reply_text(response.strip(), parse_mode='Markdown')
        
        # Дополнительная статистика для продвинутых пользователей
        try:
            model_stats = self.advanced_ai.get_model_stats()
            advanced_info = f"""🔬 *Advanced ML Statistics:*
• Модель RF: {ml_prediction.get('individual_predictions', {}).get('rf', 'N/A')}
• Модель GB: {ml_prediction.get('individual_predictions', {}).get('gb', 'N/A')}
• Модель NN: {ml_prediction.get('individual_predictions', {}).get('nn', 'N/A')}
• Общая точность: {model_stats.get('overall_accuracy', 0)*100:.1f}%
• Производительность: {model_stats.get('model_performance', 'N/A')}
• Всего предсказаний: {model_stats.get('total_predictions', 0)}"""
            
            await update.message.reply_text(advanced_info, parse_mode='Markdown')
            
        except Exception as e:
            logger.warning(f"Не удалось отправить дополнительную статистику: {e}")
        
    except Exception as e:
        logger.error(f"Ошибка в улучшенной команде aitrader: {e}")
        error_response = f"""❌ *Ошибка Advanced AI Analysis*

📌 Возможные причины:
• Временные проблемы с данными
• Недостаточно исторических данных
• Технические неполадки

🔧 *Решения:*
• Попробуйте команду `/analyse {pair if 'pair' in locals() else 'EURUSD'}`
• Используйте `/chatai` для альтернативного анализа
• Повторите запрос через несколько минут

*Ошибка: {str(e)[:100]}..."""
        
        await update.message.reply_text(error_response, parse_mode='Markdown')
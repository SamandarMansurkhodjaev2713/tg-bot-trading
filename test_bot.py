#!/usr/bin/env python3
"""
Простой скрипт для тестирования Telegram-бота
"""

import requests
import json
import time

def test_bot():
    """Тестирование основных функций бота"""
    token = "8452389061:AAFYGLK_RJ8sVSpdR5v7oPEVpY1Wy1RffL4"
    base_url = f"https://api.telegram.org/bot{token}"
    
    try:
        # Получаем информацию о боте
        response = requests.get(f"{base_url}/getMe")
        bot_info = response.json()
        
        if bot_info.get("ok"):
            print("✅ Бот успешно подключен к Telegram!")
            print(f"🤖 Имя бота: {bot_info['result']['first_name']}")
            print(f"📋 Username: @{bot_info['result']['username']}")
            
            # Получаем обновления
            response = requests.get(f"{base_url}/getUpdates")
            updates = response.json()
            
            if updates.get("ok") and updates.get("result"):
                print(f"📨 Найдено {len(updates['result'])} новых сообщений")
                
                # Отвечаем на последнее сообщение
                if updates["result"]:
                    last_update = updates["result"][-1]
                    chat_id = last_update["message"]["chat"]["id"]
                    message_text = last_update["message"].get("text", "")
                    
                    print(f"💬 Последнее сообщение: {message_text}")
                    
                    # Отправляем тестовый ответ
                    test_response = "🤖 Бот работает! Используйте команды:\n"
                    test_response += "• /start - Начать работу\n"
                    test_response += "• /analyze XAUUSD 1h - Анализ золота\n"
                    test_response += "• /news EURUSD - Новости по EUR/USD\n"
                    test_response += "• /status - Статус бота"
                    
                    response = requests.post(
                        f"{base_url}/sendMessage",
                        data={
                            "chat_id": chat_id,
                            "text": test_response,
                            "parse_mode": "Markdown"
                        }
                    )
                    
                    if response.json().get("ok"):
                        print("✅ Тестовое сообщение отправлено!")
                    else:
                        print("❌ Ошибка при отправке тестового сообщения")
                        
            else:
                print("📭 Новых сообщений не найдено")
                
        else:
            print("❌ Ошибка подключения к Telegram")
            print(f"Ошибка: {bot_info.get('description', 'Неизвестная ошибка')}")
            
    except Exception as e:
        print(f"❌ Ошибка при тестировании бота: {e}")

if __name__ == "__main__":
    print("🚀 Запуск тестирования Forex AI Advisor...")
    test_bot()
    print("✅ Тестирование завершено!")
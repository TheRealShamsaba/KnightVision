import os
import requests

def send_telegram_message(message, parse_mode="Markdown"):
    """
    Sends a message to a Telegram bot using credentials stored in environment variables.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to be set in .env or system environment.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        print("⚠️ Telegram credentials not found in environment.")
        return
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": parse_mode
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Telegram message: {e}")
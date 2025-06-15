import os
import requests
from dotenv import load_dotenv
load_dotenv('/content/drive/MyDrive/KnightVision/.env')

def send_telegram_message(message, parse_mode="Markdown"):
    """
    Sends a message to a Telegram bot using credentials stored in environment variables.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to be set in .env or system environment.
    """
    if not message or not str(message).strip():
        print("⚠️ Skipping Telegram message: empty content")
        return

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
        print("✅ Telegram API response:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Telegram message: {e}")


# --- New function for sending self-play game reports ---
def send_game_report(game_number, result, moves):
    """
    Sends a formatted message to Telegram with self-play game results.
    """
    if not result or not moves:
        print(f"⚠️ Skipping game report for Game {game_number} — missing result or moves")
        return

    result = str(result)

    message = f"♟️ *Self-Play Game {game_number}* Completed\n"
    message += f"Result: *{result}*\n"
    if len(moves) > 300:
        moves = moves[:300] + "... (truncated)"
    message += f"Moves: `{moves}`"

    if not message.strip():
        print(f"⚠️ Skipping Telegram game report for Game {game_number} — message is empty")
        return

    print("📤 Telegram game report:", message)  # Colab logging
    send_telegram_message(message)
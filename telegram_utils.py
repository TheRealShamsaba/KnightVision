import os
import requests

def send_telegram_message(message, parse_mode="HTML"):
    """
    Sends a message to a Telegram bot using credentials stored in environment variables.
    Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to be set in .env or system environment.
    """
    if not message or not str(message).strip():
        print("⚠️ Skipping Telegram message: empty content")
        return

    print("📤 Sending Telegram message:", message)

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
    
    print(f"📦 Preparing to send Telegram message:\n{message}")
    print(f"📨 Payload: {payload}")
    print(f"🔗 URL: {url}")
    
    try:
       response = requests.post(url, data=payload)
       print(f"✅ Telegram message sent: {message}")
       print(f"📬 Telegram API response: {response.status_code} - {response.text}")
       response.raise_for_status()
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

    print(f"🧩 Game Report Details — Game #{game_number}")
    print(f"Result: {result}")
    print(f"Moves: {moves}")

    message = f"♟️ *Self-Play Game {game_number}* Completed\n"
    message += f"Result: *{result}*\n"
    message += f"Moves: `{moves}`"

    print("📤 Telegram game report:", message)  # Colab logging
    send_telegram_message(message)
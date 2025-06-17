import os
import time
import requests

# Global settings controlled via environment variables
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "1").lower() not in ("0", "false")
TELEGRAM_NOTIFY_INTERVAL = int(os.getenv("TELEGRAM_NOTIFY_INTERVAL", "0"))
_last_sent = 0

def send_telegram_message(message, parse_mode="HTML", force=False):
    """Send a Telegram message if credentials are configured.

    Set ``TELEGRAM_ENABLED=0`` to disable notifications entirely. ``TELEGRAM_NOTIFY_INTERVAL``
    defines the minimum number of seconds between messages unless ``force`` is True.
    ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` must also be set.
    """


    global _last_sent

    if not TELEGRAM_ENABLED:
        print("⚠️ Telegram notifications disabled.")
        return

    if TELEGRAM_NOTIFY_INTERVAL > 0 and not force:
        now = time.time()
        if now - _last_sent < TELEGRAM_NOTIFY_INTERVAL:
            print("⏳ Skipping Telegram message due to interval limit")
            return
        _last_sent = now


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
        "parse_mode": parse_mode,
    }

    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print(f"✅ Telegram message sent: {message}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to send Telegram message: {e}")


def send_game_report(game_number, result, moves):
    """Send a formatted report summarizing a self-play game."""
    if not result or not moves:
        print(f"⚠️ Skipping game report for Game {game_number} — missing result or moves")
        return

    message = (
        f"♟️ *Self-Play Game {game_number}* Completed\n"
        f"Result: *{result}*\n"
        f"Moves: `{moves}`"
    )
    send_telegram_message(message)

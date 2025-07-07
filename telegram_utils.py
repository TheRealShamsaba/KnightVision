import os
import time
import requests
import json
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global settings controlled via environment variables
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "1").lower() not in ("0", "false")
TELEGRAM_NOTIFY_INTERVAL = int(os.getenv("TELEGRAM_NOTIFY_INTERVAL", "0"))
_last_sent = 0
SUBSCRIBERS_FILE = os.path.join(os.path.dirname(__file__), "subscribers.json")

def load_subscribers():
    try:
        with open(SUBSCRIBERS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_subscribers(ids):
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(ids, f)

def subscribe_user(chat_id):
    subs = load_subscribers()
    if chat_id not in subs:
        subs.append(chat_id)
        save_subscribers(subs)
        logger.info(f"Subscribed chat_id {chat_id}")
        return True
    return False

def unsubscribe_user(chat_id):
    subs = load_subscribers()
    if chat_id in subs:
        subs.remove(chat_id)
        save_subscribers(subs)
        logger.info(f"Unsubscribed chat_id {chat_id}")
        return True
    return False

def list_subscribers():
    return load_subscribers()

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

    # ensure subscriber list updated
    if subscribe_user(chat_id):
        logger.info(f"New subscriber added: {chat_id}")
    subscribers = list_subscribers()

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "text": message,
        "parse_mode": parse_mode,
    }

    for chat_id in subscribers:
        payload["chat_id"] = chat_id
        try:
            response = requests.post(url, data=payload)
            response.raise_for_status()
            print(f"✅ Telegram message sent to {chat_id}: {message}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to send Telegram message to {chat_id}: {e}")


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

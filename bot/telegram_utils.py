import os
import json
import logging
from datetime import datetime
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment vars
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BASE_DIR = os.getenv("BASE_DIR", ".")
STATUS_FILE = os.path.join(BASE_DIR, "last_status.json")
SUBSCRIBERS_FILE = os.path.join(BASE_DIR, "subscribers.json")
TENSORBOARD_URL = os.getenv("TENSORBOARD_URL", "")

# Helper functions for subscribers
def load_subscribers():
    try:
        with open(SUBSCRIBERS_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_subscribers(subs):
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(subs, f)

def subscribe_user(chat_id):
    subs = load_subscribers()
    if chat_id not in subs:
        subs.append(chat_id)
        save_subscribers(subs)
        logger.info(f"Subscribed {chat_id}")
    return subs

def unsubscribe_user(chat_id):
    subs = load_subscribers()
    if chat_id in subs:
        subs.remove(chat_id)
        save_subscribers(subs)
        logger.info(f"Unsubscribed {chat_id}")
    return subs

# Add list_subscribers function
def list_subscribers():
    """Return the list of subscribed chat IDs."""
    return load_subscribers()


# Helper to send a message to all subscribers
def send_telegram_message(message: str, parse_mode="HTML"):
    """
    Broadcast a message to all subscribers via Telegram Bot API.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.warning("TELEGRAM_BOT_TOKEN not set; cannot send message.")
        return
    subs = load_subscribers()
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in subs:
        payload = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }
        try:
            requests.post(url, data=payload, timeout=5)
        except Exception as e:
            logger.error(f"Error sending message to {chat_id}: {e}")

# Command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    subscribe_user(chat_id)
    await update.message.reply_text("✅ Subscribed to training updates. Use /stop to unsubscribe.")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    unsubscribe_user(chat_id)
    await update.message.reply_text("🚫 Unsubscribed from training updates. Use /start to subscribe again.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                s = json.load(f)
            text = (
                f"🏁 *Last Training Status:*\n"
                f"Epoch: `{s.get('epoch', '?')}`\n"
                f"Train Loss: `{s.get('train_loss', '?')}`\n"
                f"Accuracy: `{s.get('accuracy', '?')*100:.2f}%`\n"
                f"Avg Reward: `{s.get('avg_reward', '?')}`\n"
                f"Score: `{s.get('score', '?')}`\n"
                f"Timestamp: `{s.get('timestamp', '?')}`"
            )
        except Exception as e:
            text = f"❌ Error reading status file: {e}"
    else:
        text = "⚠️ No status file found."
    await update.message.reply_markdown(text)

async def graphs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if TENSORBOARD_URL:
        await update.message.reply_text(f"📊 TensorBoard: {TENSORBOARD_URL}")
    else:
        await update.message.reply_text("⚠️ TensorBoard URL not configured.")

async def relay(update: Update, context: ContextTypes.DEFAULT_TYPE):
    admin_id = os.getenv("ADMIN_CHAT_ID")
    if str(update.effective_chat.id) != str(admin_id):
        return
    msg = " ".join(context.args)
    if msg:
        # broadcast
        for cid in load_subscribers():
            await context.bot.send_message(chat_id=cid, text=msg)
        await update.message.reply_text("📨 Message broadcast.")
    else:
        await update.message.reply_text("Usage: /relay <message>")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cmds = [
        "/start - Subscribe to updates",
        "/stop - Unsubscribe",
        "/status - Show last training status",
        "/graphs - Get TensorBoard link",
        "/relay - Admin broadcast",
        "/help - This help"
    ]
    await update.message.reply_text("\n".join(cmds))

def main():
    if not TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return
    app = ApplicationBuilder().token(TOKEN).build()
    # Register handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("graphs", graphs))
    app.add_handler(CommandHandler("relay", relay))
    app.add_handler(CommandHandler("help", help_command))
    logger.info("Bot starting...")
    app.run_polling()

if __name__ == "__main__":
    main()

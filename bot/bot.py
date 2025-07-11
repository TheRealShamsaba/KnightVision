#!/usr/bin/env python3
import os
import json
import logging
from datetime import datetime

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackContext

from bot.telegram_utils import (
    subscribe_user,
    unsubscribe_user,
    list_subscribers,
    send_telegram_message
)

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Where training writes its last‐status JSON
BASE_DIR = os.getenv("BASE_DIR", "/content/drive/MyDrive/KnightVision")
STATUS_FILE = os.path.join(BASE_DIR, "last_status.json")
# Optional: your TensorBoard URL if you host it somewhere
TENSORBOARD_URL = os.getenv("TENSORBOARD_URL", "")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    logger.error("Please set TELEGRAM_BOT_TOKEN in env")
    exit(1)

def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    if subscribe_user(chat_id):
        update.message.reply_text("✅ Subscribed! You’ll now get training updates.")
    else:
        update.message.reply_text("👍 You were already subscribed.")

def stop(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    if unsubscribe_user(chat_id):
        update.message.reply_text("🚫 Unsubscribed. You will no longer receive updates.")
    else:
        update.message.reply_text("ℹ️ You weren’t subscribed.")

def status(update: Update, context: CallbackContext):
    """Show the last training metrics."""
    if os.path.exists(STATUS_FILE):
        try:
            with open(STATUS_FILE, "r") as f:
                s = json.load(f)
            text = (
                f"🏁 *Last Training Run*\n"
                f"• Epoch: `{s['epoch']}`\n"
                f"• Train Loss: `{s['train_loss']:.4f}`\n"
                f"• Val Loss: `{s['val_loss']:.4f}`\n"
                f"• Accuracy: `{s['accuracy']*100:.2f}%`\n"
                f"• Timestamp: `{s['timestamp']}`"
            )
        except Exception as e:
            text = f"❌ Failed to read status: {e}"
    else:
        text = "⚠️ No training status found yet."
    update.message.reply_markdown(text)

def graphs(update: Update, context: CallbackContext):
    """Send a link to TensorBoard or inform if unavailable."""
    if TENSORBOARD_URL:
        update.message.reply_text(f"📊 View full TensorBoard metrics here:\n{TENSORBOARD_URL}")
    else:
        update.message.reply_text("⚠️ TensorBoard URL not configured. Set TENSORBOARD_URL in env.")

def relay(update: Update, context: CallbackContext):
    """A helper command so admins can broadcast a message to all subscribers."""
    admin_id = os.getenv("ADMIN_CHAT_ID")
    if update.effective_chat.id != int(admin_id):
        return
    msg = " ".join(context.args)
    if not msg:
        update.message.reply_text("Usage: /relay <your message>")
    else:
        send_telegram_message(msg)
        update.message.reply_text("📨 Message broadcast to all subscribers.")

def main():
    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("graphs", graphs))
    app.add_handler(CommandHandler("relay", relay))

    logger.info("Bot starting…")
    app.run_polling()

if __name__ == "__main__":
    main()
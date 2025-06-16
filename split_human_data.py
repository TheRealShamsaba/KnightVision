import sys
from pathlib import Path
import os
import requests
from dotenv import load_dotenv
load_dotenv()

if "google.colab" in sys.modules:
    BASE_DIR = "/content/drive/MyDrive/KnightVision/basicChess"
else:
    BASE_DIR = str(Path(__file__).resolve().parent)
input_path = "/content/drive/MyDrive/KnightVision/data/games.jsonl"
output_dir = os.path.join(BASE_DIR, "data/human_batches")
lines_per_file = 100000  # 100,000 lines per file

def notify_bot(message):
    if str(os.getenv("ENABLE_TELEGRAM", "true")).lower() in ("false", "0", "no"):
        print("📵 Telegram disabled via ENABLE_TELEGRAM. Skipping message.")
        return

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ Telegram credentials not set in environment.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": chat_id, "text": message})
    except Exception as e:
        print(f"❌ Failed to send message to bot: {e}")

os.makedirs(output_dir, exist_ok=True)

def split_file(input_file, output_directory, chunk_size):
    notify_bot("📤 Starting human data split...")
    with open(input_file, "r") as infile:
        file_count = 0
        line_count = 0
        outfile = open(os.path.join(output_directory, f"games_part_{file_count}.jsonl"), "w")
        for i, line in enumerate(infile):
            if i > 0 and i % chunk_size == 0:
                outfile.close()
                print(f"✅ Saved games_part_{file_count}.jsonl with {chunk_size} lines")
                notify_bot(f"✅ Saved games_part_{file_count}.jsonl with {chunk_size if i % chunk_size == 0 else 'remaining'} lines")
                file_count += 1
                outfile = open(os.path.join(output_directory, f"games_part_{file_count}.jsonl"), "w")
            outfile.write(line)
            line_count += 1
        outfile.close()
        print(f"✅ Saved games_part_{file_count}.jsonl with remaining lines")
        notify_bot(f"✅ Saved games_part_{file_count}.jsonl with {chunk_size if i % chunk_size == 0 else 'remaining'} lines")
        print(f"🎉 Split complete. Total lines: {line_count}, Total files: {file_count + 1}")
        notify_bot(f"🎉 Human data split complete. Total lines: {line_count}, Total files: {file_count + 1}")

try:
    split_file(input_path, output_dir, lines_per_file)
except Exception as e:
    error_message = f"❌ Error during data split: {str(e)}"
    print(error_message)
    notify_bot(error_message)
import os
import requests
import logging
from dotenv import load_dotenv
from logging_utils import configure_logging
load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

def send_telegram_message(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.error("❌ Telegram credentials not set in environment.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        logger.error("❌ Failed to send Telegram message: %s", e)
BASE_DIR = "/content/drive/MyDrive/KnightVision/data"

import chess.pgn
import chess
import json

PARSED_LOG = os.path.join(BASE_DIR, "parsed_files.log")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def notify_bot(message):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message}
        )
    except Exception as e:
        logger.error("❌ Failed to send message to bot: %s", e)

def get_parsed_files():
    if os.path.exists(PARSED_LOG):
        with open(PARSED_LOG, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def mark_file_parsed(filename):
    with open(PARSED_LOG, 'a') as f:
        f.write(filename + "\n")

def extract_data_from_pgn(pgn_path):
    count = 0
    try:
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                board = game.board()
                result = game.headers.get("Result", "*")
                if result == "1-0":
                    outcome = 1
                elif result == "0-1":
                    outcome = -1
                elif result == "1/2-1/2":
                    outcome = 0
                else:
                    outcome = None  # unknown or ongoing game

                for move in game.mainline_moves():
                    fen = board.fen()
                    san = board.san(move)
                    board.push(move)
                    yield {"fen": fen, "move": san, "outcome": outcome}
                    count += 1
                    if count % 100000 == 0:
                        logger.info("🕹️ Parsed %s moves so far...", f"{count:,}")
                        notify_bot(f"🕹️ Parsed {count:,} moves so far from {pgn_path}")

    except Exception as e:
        logger.error("Failed to parse %s: %s", pgn_path, e)

def parse_all_games(pgn_dir=os.path.join(BASE_DIR, "data", "pgn"), output_path=os.path.join(BASE_DIR, "data", "games.jsonl")):
    if not os.path.exists(pgn_dir):
        notify_bot(f"❌ PGN directory not found: {pgn_dir}")
        logger.error("❌ PGN directory not found: %s", pgn_dir)
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    parsed_files = get_parsed_files()
    with open(output_path, 'a', encoding='utf-8') as out_file:
        for filename in os.listdir(pgn_dir):
            if filename.endswith(".pgn") and filename not in parsed_files:
                logger.info("Parsing %s...", filename)
                notify_bot(f"🤖 Starting to parse: {filename}")
                pgn_path = os.path.join(pgn_dir, filename)
                count = 0
                for record in extract_data_from_pgn(pgn_path):
                    out_file.write(json.dumps(record) + "\n")
                    count += 1
                mark_file_parsed(filename)
                logger.info("✅ Finished parsing %s", filename)
                notify_bot(f"✅ Done parsing {filename} ({count} moves)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse PGN files")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level")
    args = parser.parse_args()
    configure_logging(args.log_level)

    parse_all_games(
        pgn_dir=os.path.join(BASE_DIR, "pgn"),
        output_path=os.path.join(BASE_DIR, "games.jsonl")
    )

# Reminder: Set the required environment variables before running this script,
# or create a .env file with the following contents:
# TELEGRAM_BOT_TOKEN=your_bot_token
# TELEGRAM_CHAT_ID=your_chat_id

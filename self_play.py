# self_play.py
import os
import time
import numpy as np
from telegram_utils import send_telegram_message
import logging
from logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
logger.info("Self-play script loaded...")
logger.info("✅ Telegram test message dispatched.")
logger.info("📋 Note: All Telegram messages will log their intent before sending.")

try:
    send_telegram_message("📥 self_play.py loaded successfully.")
except Exception as e:
    logger.error("⚠️ Telegram send failed: %s", e)

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=True)
        BASE_DIR = "/content/drive/MyDrive/KnightVision"
    except Exception as e:
        logger.error("⚠️ Colab drive mount failed: %s", e)
        BASE_DIR = os.getenv("BASE_DIR", "/content/drive/MyDrive/KnightVision")
else:
    logger.info("📦 Not running in Colab — skipping drive.mount")
    from dotenv import load_dotenv
    load_dotenv()
    BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chessEngine import GameState
from ai import encode_board, encode_move
import random
import torch
import logging
from model import ChessNet
import psutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("🖥️ Using device: %s", device)

if torch.cuda.is_available():
    logger.info("💾 VRAM used: %.2f MB", torch.cuda.memory_allocated(device) / 1024 ** 2)

logger = logging.getLogger(__name__)


def self_play(model, num_games=100, device=None):
    logger.info("✅ self_play() function has started executing")
    data = []
    model.eval()
    model.to(device)
    model = model.float()
    try:
        send_telegram_message(f"🤖 Starting self-play with {num_games} games...")
    except Exception as e:
        logger.error("⚠️ Telegram send failed: %s", e)
    logger.info("Starting self-play with %s games...", num_games)
    logger.debug("🧪 Self-play loop entered")
    try:
        send_telegram_message(f"🎮 Confirmed: Self-play function is running with {num_games} games.")
    except Exception as e:
        logger.error("⚠️ Telegram send failed: %s", e)
    for _ in range(num_games):
        logger.info("🕹️ Starting game %s/%s", _ + 1, num_games)
        logger.debug("⏳ Game initialization complete — entering move loop")
        gs = GameState()
        game_data = []
        MAX_MOVES = 200  # hard limit to prevent endless games
        move_count = 0

        while move_count < MAX_MOVES:  # Continue until the game ends naturally or max moves reached
            valid_moves = gs.getValidMoves()
            logger.debug("♟️ Valid moves count: %s", len(valid_moves))
            if not valid_moves:
                break

            encoded = np.array([encode_board(gs.board)])
            encoded = encoded.astype(np.float32)
            board_tensor = torch.from_numpy(encoded).float().to(device)
            with torch.no_grad():
                policy_logits, _ = model(board_tensor)
            policy = torch.softmax(policy_logits.squeeze(), dim=0).detach().cpu().numpy()

            legal_indices = [encode_move(m.startRow, m.startCol, m.endRow, m.endCol) for m in valid_moves]
            legal_probs = [policy[i] if i < len(policy) else 0 for i in legal_indices]

            total_weight = sum(legal_probs)
            if total_weight == 0:
                move = random.choice(valid_moves)
            else:
                normalized = [w / total_weight for w in legal_probs]
                move = random.choices(valid_moves, weights=normalized, k=1)[0]

            move_index = encode_move(move.startRow, move.startCol, move.endRow, move.endCol)
            game_data.append((encode_board(gs.board), move_index))
            if len(game_data) % 10 == 0:
                logger.debug("♟️ Played %s moves so far...", len(game_data))
            gs.makeMove(move)
            move_count += 1

            time.sleep(0.01)

        if move_count >= MAX_MOVES:
            result_reason = "Max move limit reached"
            outcome = 0.0  # can tune this if needed
        else:
            # Assign outcome based on game end state
            if gs.inCheck() and len(gs.getValidMoves()) == 0:
                # Checkmate
                outcome = 1 if not gs.whiteToMove else -1
                result_reason = "Checkmate"
            elif len(gs.getValidMoves()) == 0:
                # Stalemate
                outcome = 0.5
                result_reason = "Stalemate"
            elif gs.isDraw():
                # Draw by repetition or 50-move rule
                outcome = 0.5
                result_reason = "Draw (50-move or repetition)"
            else:
                # Game ended early — reward based on material balance
                white_material = sum(piece_value(p) for r in gs.board for p in r if p.isupper())
                black_material = sum(piece_value(p) for r in gs.board for p in r if p.islower())
                if white_material == black_material:
                    outcome = 0
                else:
                    outcome = (white_material - black_material) / max(white_material, black_material)
                result_reason = "Material difference"
        for state, move in game_data:
            data.append((state, move, outcome))
        message = f"🏁 Game finished — {result_reason}. Moves: {len(game_data)} | Outcome: {outcome}"
        if outcome == 0.5:
            try:
                send_telegram_message("⚖️ Draw detected during self-play.")
            except Exception as e:
                logger.error("⚠️ Telegram send failed: %s", e)
        logger.debug("📨 Message to send: %s", message)
        try:
            send_telegram_message(message)
        except Exception as e:
            logger.error("⚠️ Telegram send failed: %s", e)

        if game_data:
            sample = game_data[0]
            try:
                send_telegram_message(f"🎯 Sample game generated.\nMoves: {len(game_data)} | First move index: {sample[1]}")
            except Exception as e:
                logger.error("⚠️ Telegram send failed: %s", e)
        else:
            logger.warning("⚠️ No game data generated to report.")
        logger.debug("🧠 RAM usage: %s%%", psutil.virtual_memory().percent)
        logger.info("✅ Game %s complete. Moves played: %s | Outcome: %s", _ + 1, len(game_data), outcome)
        if torch.cuda.is_available():
            logger.debug("💾 VRAM: %.2f MB", torch.cuda.memory_allocated(device) / 1024 ** 2)

    return data


# Piece value function for material evaluation
def piece_value(piece):
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    return values.get(piece.upper(), 0)


def generate_self_play_data(model, num_games=50, device=None):
    return self_play(model, num_games, device)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run self-play")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level")
    args = parser.parse_args()
    configure_logging(args.log_level)

    model = ChessNet()
    model_path = os.path.join(BASE_DIR, "checkpoints", "model.pth")
    if not os.path.exists(model_path):
        logger.error("❌ Model checkpoint not found: %s", model_path)
        exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("✅ Model loaded successfully — starting self_play()")
    try:
        send_telegram_message("📦 Self-play started from __main__ with loaded model.")
    except Exception as e:
        logger.error("⚠️ Telegram send failed: %s", e)
    model.to(device)
    logger.info("✅ Loaded model from %s", model_path)
    data = self_play(model, num_games=50, device=device)

    import json
    import numpy as np

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    save_path = os.path.join(BASE_DIR, f"self_play_data_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
    with open(save_path, "w") as f:
        for state, move, outcome in data:
            f.write(json.dumps({
                "state": state,
                "move": move,
                "outcome": outcome
            }, cls=NumpyEncoder) + "\n")
    logger.info("💾 Saved self-play data to %s — %s samples.", save_path, len(data))

    logger.info("🧪 self_play() execution finished.")
    logger.info("Generated %s samples from self-play", len(data))

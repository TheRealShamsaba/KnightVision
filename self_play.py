# self_play.py
import os
import time
from telegram_utils import send_telegram_message
print("Self-play script loaded...")

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
        print(f"⚠️ Colab drive mount failed: {e}")
        BASE_DIR = os.getenv("BASE_DIR", "/content/drive/MyDrive/KnightVision")
else:
    print("📦 Not running in Colab — skipping drive.mount")
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
print(f"🖥️ Using device: {device}")

if torch.cuda.is_available():
    print(f"💾 VRAM usage: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB / {torch.cuda.max_memory_allocated(device) / 1024 ** 2:.2f} MB")

logger = logging.getLogger(__name__)


def self_play(model, num_games=100):
    data = []
    model.eval()
    model.to(device)
    send_telegram_message(f"🤖 Starting self-play with {num_games} games...")
    print(f"Starting self-play with {num_games} games...")
    for _ in range(num_games):
        gs = GameState()
        game_data = []

        for _ in range(100):  # max moves per game
            valid_moves = gs.getValidMoves()
            if not valid_moves:
                break

            board_tensor = torch.tensor([encode_board(gs.board)]).float().to(device)
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
            gs.makeMove(move)

            time.sleep(0.01)

        # Assign outcome based on game end state
        if gs.inCheck() and len(gs.getValidMoves()) == 0:
            # Checkmate
            outcome = 1 if not gs.whiteToMove else -1
        elif len(gs.getValidMoves()) == 0:
            # Stalemate
            outcome = 0.5
        elif gs.isDraw():
            # Draw by repetition or insufficient material (you may need to implement this)
            outcome = 0.5
        else:
            # Game ended early — reward based on material balance
            white_material = sum(piece_value(p) for r in gs.board for p in r if p.isupper())
            black_material = sum(piece_value(p) for r in gs.board for p in r if p.islower())
            if white_material == black_material:
                outcome = 0
            else:
                outcome = (white_material - black_material) / max(white_material, black_material)
        for state, move_index in game_data:
            data.append((state, move_index, outcome))
        if len(game_data) > 10:
            sample = game_data[0]
            send_telegram_message(f"🎯 Sample game generated with {len(game_data)} moves.\nFirst move index: {sample[1]}")
        print(f"🧠 RAM usage: {psutil.virtual_memory().percent}%")
        if torch.cuda.is_available():
            print(f"💾 VRAM: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f} MB")

    send_telegram_message(f"✅ Self-play completed. {len(data)} samples generated.")
    return data


# Piece value function for material evaluation
def piece_value(piece):
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    return values.get(piece.upper(), 0)


if __name__ == "__main__":
    model = ChessNet()
    model_path = os.path.join(BASE_DIR, "checkpoints", "model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    send_telegram_message("📦 Self-play started from __main__ with loaded model.")
    model.to(device)
    print(f"✅ Loaded model from {model_path}")
    data = self_play(model, num_games=50)
    logger.info("Generated %s samples from self-play", len(data))
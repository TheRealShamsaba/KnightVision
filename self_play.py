# self_play.py
import os
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
        IN_COLAB = False
        BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
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

logger = logging.getLogger(__name__)


def self_play(model, num_games=100):
    data = []
    model.eval()
    print(f"Starting self-play with {num_games} games...")
    for _ in range(num_games):
        gs = GameState()
        game_data = []

        for _ in range(100):  # max moves per game
            valid_moves = gs.getValidMoves()
            if not valid_moves:
                break

            board_tensor = torch.tensor([encode_board(gs.board)]).float()
            with torch.no_grad():
                policy_logits, _ = model(board_tensor)
            policy = torch.softmax(policy_logits.squeeze(), dim = 0).cpu().numpy()

            legal_indices = [encode_move(m.startRow, m.startCol, m.endRow, m.endCol) for m in valid_moves]
            legal_probs = [policy[i] for i in legal_indices]

            total_weight = sum(legal_probs)
            if total_weight == 0:
                move = random.choice(valid_moves)
            else:
                normalized = [w / total_weight for w in legal_probs]
                move = random.choices(valid_moves, weights=normalized, k=1)[0]

            move_index = encode_move(move.startRow, move.startCol, move.endRow, move.endCol)
            game_data.append((encode_board(gs.board), move_index))
            gs.makeMove(move)

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

    return data


# Piece value function for material evaluation
def piece_value(piece):
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    return values.get(piece.upper(), 0)


if __name__ == "__main__":
    model = ChessNet()
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "checkpoints", "model.pth")))  # Replace with your model file
    data = self_play(model, num_games=50)
    logger.info("Generated %s samples from self-play", len(data))
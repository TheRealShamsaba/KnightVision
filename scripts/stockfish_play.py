import subprocess
import chess
import chess.engine
import torch
from ai.model import ChessNet
from ai import encode_board, legal_move_indices
import random
import numpy as np
import tensorflow as tf
import os
import datetime

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
tf.random.set_seed(SEED)

# Setup TF summary writer for Stockfish evaluation
LOG_DIR = os.path.join("runs", "stockfish_eval", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(LOG_DIR, exist_ok=True)
tf_writer = tf.summary.create_file_writer(LOG_DIR)

def play_vs_stockfish(model, num_games=10, stockfish_path="/usr/games/stockfish", skill_level=5, max_moves=250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    if device.type != "cuda":
        print("⚠️ WARNING: Running on CPU — Stockfish eval will be slower.")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": skill_level})

    results = {"win": 0, "loss": 0, "draw": 0}

    # Log initial tallies
    with tf_writer.as_default():
        tf.summary.scalar("Stockfish/wins", 0, step=0)
        tf.summary.scalar("Stockfish/losses", 0, step=0)
        tf.summary.scalar("Stockfish/draws", 0, step=0)

    try:
        for game_num in range(num_games):
            print(f"🎮 Starting Game {game_num + 1}/{num_games}")
            board = chess.Board()
            move_count = 0

            ai_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK

            while not board.is_game_over() and move_count < max_moves:
                if board.turn == ai_color:
                    # AI's turn: encode board and get network logits
                    encoded = encode_board(board)
                    board_np = np.expand_dims(encoded, axis=0)  # shape (1, ...)
                    board_tensor = torch.from_numpy(board_np).float().to(device)
                    
                    with torch.no_grad():
                        logits, _ = model(board_tensor)  # shape [1, 4096]
                        # Only consider logits of legal moves
                        legal = legal_move_indices(board)
                        if not legal:
                            # no legal moves: fall back
                            result = engine.play(board, chess.engine.Limit(time=0.01))
                            board.push(result.move)
                        else:
                            mask = torch.zeros_like(logits, dtype=torch.bool)
                            mask[0, list(legal)] = True
                            filtered = logits.masked_fill(~mask, float('-inf'))
                            move_idx = torch.argmax(filtered, dim=1).item()
                            board.push(legal[move_idx])
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)

                move_count += 1

            outcome = board.result()
            print(f"🧾 Game Result: {outcome}")
            if (outcome == "1-0" and ai_color == chess.WHITE) or (outcome == "0-1" and ai_color == chess.BLACK):
                results["win"] += 1
            elif (outcome == "1-0" and ai_color == chess.BLACK) or (outcome == "0-1" and ai_color == chess.WHITE):
                results["loss"] += 1
            else:
                results["draw"] += 1

            # Update TF summary after each game
            with tf_writer.as_default():
                tf.summary.scalar("Stockfish/wins", results["win"], step=game_num+1)
                tf.summary.scalar("Stockfish/losses", results["loss"], step=game_num+1)
                tf.summary.scalar("Stockfish/draws", results["draw"], step=game_num+1)

    finally:
        engine.quit()
    tf_writer.flush()
    print("✅ Finished Stockfish Evaluation")
    print("Summary:", results)
    return results


if __name__ == "__main__":
    import argparse
    import torch

    parser = argparse.ArgumentParser(description="Play KnightVision model against Stockfish")
    parser.add_argument("--model-path",     required=True, help="Path to the .pth model checkpoint")
    parser.add_argument("--num-games",      type=int,   default=10, help="Number of games to play")
    parser.add_argument("--stockfish-path", default="/usr/games/stockfish", help="Path to Stockfish binary")
    parser.add_argument("--skill-level",    type=int,   default=5, help="Stockfish skill level")
    parser.add_argument("--max-moves",      type=int,   default=250, help="Max half-moves per game")
    args = parser.parse_args()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessNet().to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    # Run evaluation
    play_vs_stockfish(
        model,
        num_games      = args.num_games,
        stockfish_path = args.stockfish_path,
        skill_level    = args.skill_level,
        max_moves      = args.max_moves,
    )
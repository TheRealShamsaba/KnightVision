import subprocess
import chess
import chess.engine
import torch
from model import ChessNet
from ai import encode_board
from ai import decode_move_index

def play_vs_stockfish(model, num_games=10, stockfish_path="/usr/games/stockfish", skill_level=5, max_moves=250):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model = model.to(device)

    if device.type != "cuda":
        print("⚠️ WARNING: Running on CPU — Stockfish eval will be slower.")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Skill Level": skill_level})

    results = {"win": 0, "loss": 0, "draw": 0}

    try:
        for game_num in range(num_games):
            print(f"🎮 Starting Game {game_num + 1}/{num_games}")
            board = chess.Board()
            move_count = 0

            ai_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK

            while not board.is_game_over() and move_count < max_moves:
                if board.turn == ai_color:
                    if not isinstance(board, chess.Board):
                        raise ValueError(f"Expected chess.Board, got {type(board)}")
                    encoded = encode_board(board)
                    board_tensor = torch.tensor([encoded], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        policy_logits, _ = model(board_tensor)
                        move_idx = torch.argmax(policy_logits, dim=1).item()

                    start_row, start_col, end_row, end_col = decode_move_index(move_idx)
                    move = chess.Move.from_uci(f"{chr(start_col + 97)}{8 - start_row}{chr(end_col + 97)}{8 - end_row}")
                    
                    # If the AI predicts an illegal move, fall back to Stockfish for a legal move
                    if move not in board.legal_moves:
                        print(f"⚠️ Illegal move predicted: {move}. Falling back to engine.")
                        result = engine.play(board, chess.engine.Limit(time=0.01))
                        board.push(result.move)
                    else:
                        board.push(move)
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.1))
                    board.push(result.move)

                move_count += 1

            outcome = board.result()
            print(f"🧾 Game Result: {outcome}")
            if outcome == "1-0":
                results["win" if ai_color == chess.WHITE else "loss"] += 1
            elif outcome == "0-1":
                results["loss" if ai_color == chess.WHITE else "win"] += 1
            else:
                results["draw"] += 1
    finally:
        engine.quit()
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
    model.load_state_dict(state)
    model.eval()

    # Run evaluation
    play_vs_stockfish(
        model,
        num_games      = args.num_games,
        stockfish_path = args.stockfish_path,
        skill_level    = args.skill_level,
        max_moves      = args.max_moves,
    )
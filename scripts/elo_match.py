"""
Estimates KnightVision's Elo by playing it against Stockfish at calibrated strengths.

Stockfish's UCI_Elo setting is calibrated at a 60s + 0.6s time control, so Stockfish gets
that clock here. KnightVision plays its top legal policy move (one forward pass, no search).

    PYTHONPATH=. python scripts/elo_match.py --model-path Best_Model.pth --elos 1320 1500 1700 --games 40
"""
import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import chess
import chess.engine
import chess.pgn
import torch

from ai import encode_board, legal_move_indices
from ai.model import ChessNet

_model = None


def _load_model(model_path):
    global _model
    if _model is None:
        torch.set_num_threads(1)
        state = torch.load(model_path, map_location="cpu")
        _model = ChessNet()
        _model.load_state_dict(state.get("model_state_dict", state))
        _model.eval()
    return _model


def knightvision_move(model, board):
    legal = legal_move_indices(board)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(encode_board(board)).unsqueeze(0))
    best = max(legal, key=lambda index: logits[0, index].item())
    return legal[best]


def play_game(model_path, stockfish_path, elo, kv_white, base, increment, max_plies=400):
    model = _load_model(model_path)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": elo})
    board = chess.Board()
    clock = base
    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
            if (board.turn == chess.WHITE) == kv_white:
                board.push(knightvision_move(model, board))
            else:
                # Stockfish gets a real clock; KnightVision's side is given the same nominal time.
                limit = chess.engine.Limit(
                    white_clock=clock if not kv_white else base, black_clock=clock if kv_white else base,
                    white_inc=increment, black_inc=increment,
                )
                start = time.monotonic()
                board.push(engine.play(board, limit).move)
                clock = max(0.05, clock - (time.monotonic() - start) + increment)
    finally:
        engine.quit()

    result = board.result(claim_draw=True)
    if result == "*":
        result = "1/2-1/2"  # ply limit reached
    score = {"1-0": 1.0, "0-1": 0.0}.get(result, 0.5)
    kv_score = score if kv_white else 1.0 - score

    game = chess.pgn.Game.from_board(board)
    game.headers["White"] = "KnightVision" if kv_white else f"Stockfish {elo}"
    game.headers["Black"] = f"Stockfish {elo}" if kv_white else "KnightVision"
    game.headers["Result"] = result
    return {"elo": elo, "kv_white": kv_white, "result": result, "kv_score": kv_score,
            "plies": board.ply(), "pgn": str(game)}


def expected(rating, opponent):
    return 1.0 / (1.0 + 10 ** ((opponent - rating) / 400))


def estimate_rating(games):
    """
    Maximum-likelihood rating against the opponents' Elos, with a 95% interval from the
    curvature of the log-likelihood. Returns None when every game was won or lost.
    """
    total = sum(g["kv_score"] for g in games)
    if total in (0, len(games)):
        return None
    lo, hi = -1000.0, 4000.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if sum(expected(mid, g["elo"]) for g in games) < total:
            lo = mid
        else:
            hi = mid
    rating = (lo + hi) / 2
    information = sum(expected(rating, g["elo"]) * (1 - expected(rating, g["elo"])) for g in games)
    stderr = 400 / math.log(10) / math.sqrt(information)
    return rating, 1.96 * stderr


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--elos", type=int, nargs="+", default=[1320, 1500, 1700])
    parser.add_argument("--games", type=int, default=40, help="games per Elo level, colours alternate")
    parser.add_argument("--base", type=float, default=60.0, help="Stockfish base time in seconds")
    parser.add_argument("--increment", type=float, default=0.6)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--out", default=os.path.join("runs", "elo", datetime.now().strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    jobs = [(args.model_path, args.stockfish_path, elo, i % 2 == 0, args.base, args.increment)
            for elo in args.elos for i in range(args.games)]
    games = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(os.path.join(args.out, "games.jsonl"), "w") as log:
        for game in pool.map(play_game, *zip(*jobs)):
            games.append(game)
            log.write(json.dumps(game) + "\n")
            log.flush()
            print(f"vs {game['elo']}: {game['result']} as {'white' if game['kv_white'] else 'black'} "
                  f"({game['plies']} plies)", flush=True)

    summary = {"levels": {}}
    for elo in args.elos:
        level = [g for g in games if g["elo"] == elo]
        wins = sum(g["kv_score"] == 1.0 for g in level)
        draws = sum(g["kv_score"] == 0.5 for g in level)
        summary["levels"][elo] = {"games": len(level), "wins": wins, "draws": draws,
                                  "losses": len(level) - wins - draws,
                                  "score": sum(g["kv_score"] for g in level) / len(level)}
    estimate = estimate_rating(games)
    if estimate:
        summary["rating"], summary["margin95"] = round(estimate[0]), round(estimate[1])
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out, "games.pgn"), "w") as f:
        f.write("\n\n".join(g["pgn"] for g in games))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

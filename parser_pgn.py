import os
import chess.pgn
import chess
import json

PARSED_LOG = "data/parsed_files.log"

def get_parsed_files():
    if os.path.exists(PARSED_LOG):
        with open(PARSED_LOG, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def mark_file_parsed(filename):
    with open(PARSED_LOG, 'a') as f:
        f.write(filename + "\n")

def extract_data_from_pgn(pgn_path):
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
    except Exception as e:
        print(f"Failed to parse {pgn_path}: {e}")

def parse_all_games(pgn_dir="data/pgn", output_path="data/games.jsonl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    parsed_files = get_parsed_files()
    with open(output_path, 'a', encoding='utf-8') as out_file:
        for filename in os.listdir(pgn_dir):
            if filename.endswith(".pgn") and filename not in parsed_files:
                print(f"Parsing {filename}...")
                pgn_path = os.path.join(pgn_dir, filename)
                for record in extract_data_from_pgn(pgn_path):
                    out_file.write(json.dumps(record) + "\n")
                mark_file_parsed(filename)

if __name__ == "__main__":
    parse_all_games()

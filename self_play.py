# self_play.py

from chessEngine import GameState
from ai import encode_board, encode_move
import random
import torch
from model import ChessNet

def self_play(model, num_games=100):
    data = []
    model.eval()
    for _ in range(num_games):
        gs = GameState()
        game_data = []

        for _ in range(100):  # max moves per game
            valid_moves = gs.getValidMoves()
            if not valid_moves:
                break

            board_tensor = torch.tensor([encode_board(gs.board)]).float()
            with torch.no_grad():
                policy, _ = model(board_tensor)
            policy = policy.squeeze().cpu().numpy()

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
        if gs.inCheck():
            if len(gs.getValidMoves()) == 0:
                outcome = -1 if gs.whiteToMove else 1  # last player gave checkmate
            else:
                outcome = 0  # game not finished
        elif len(gs.getValidMoves()) == 0:
            outcome = 0  # stalemate
        else:
            outcome = 0  # game ended early
        for state, move_index in game_data:
            data.append((state, move_index, outcome))

    return data


if __name__ == "__main__":
    model = ChessNet()
    model.load_state_dict(torch.load("model.pth"))  # Replace with your model file
    data = self_play(model, num_games=50)
    print("Generated", len(data), "samples from self-play")
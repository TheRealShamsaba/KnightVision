import chess
from typing import Union, List

import numpy as np

# The one board/move encoding shared by training, self-play, evaluation and play.
# It matches the supervised training run that produced the existing checkpoints:
#
#   Board: 12 planes of 8x8, in PIECE_PLANES order (white P N B R Q K, then black).
#          Row 0 is rank 8 and column 0 is file a, the same layout as GameState.board.
#   Move:  from_square * 64 + to_square, with python-chess square numbers (a1 = 0, h8 = 63).
#          The promotion piece is not encoded; callers promote to a queen.

PIECE_PLANES = "PNBRQKpnbrqk"
PIECE_TO_INDEX = {symbol: i for i, symbol in enumerate(PIECE_PLANES)}
INDEX_TO_PIECE = {i: symbol for symbol, i in PIECE_TO_INDEX.items()}

def square_from_rc(row: int, col: int) -> int:
    """GameState (row, col) to a python-chess square number."""
    return (7 - row) * 8 + col

def rc_from_square(square: int):
    """python-chess square number to GameState (row, col)."""
    return 7 - square // 8, square % 8

def _engine_code_to_symbol(code: str):
    """GameState piece code ("wK", "bp", "--") to a python-chess symbol ("K", "p", None)."""
    if len(code) != 2 or code[0] not in "wb":
        return None
    letter = code[1].upper()
    return letter if code[0] == "w" else letter.lower()

def encode_board(board: Union[chess.Board, str, List[List[str]], np.ndarray]) -> np.ndarray:
    """
    Encodes a python-chess Board, a FEN string, or a GameState.board grid as a (12, 8, 8) array.
    """
    encoded = np.zeros((12, 8, 8), dtype=np.float32)

    if isinstance(board, str):
        board = chess.Board(board)

    if isinstance(board, chess.Board):
        for square, piece in board.piece_map().items():
            row, col = rc_from_square(square)
            encoded[PIECE_TO_INDEX[piece.symbol()], row, col] = 1.0
        return encoded

    for row in range(8):
        for col in range(8):
            symbol = _engine_code_to_symbol(str(board[row][col]))
            if symbol is not None:
                encoded[PIECE_TO_INDEX[symbol], row, col] = 1.0
    return encoded

def encode_move(start_row, start_col, end_row, end_col):
    """
    Converts a GameState move (rows counted from rank 8) into a flat index (0-4095).
    """
    return square_from_rc(start_row, start_col) * 64 + square_from_rc(end_row, end_col)

def encode_chess_move(move: chess.Move) -> int:
    """
    Converts a python-chess move into a flat index (0-4095).
    """
    return move.from_square * 64 + move.to_square

def decode_move_index(index):
    """
    Converts a flat index (0-4095) back to a GameState move (start_row, start_col, end_row, end_col).
    """
    start_row, start_col = rc_from_square(index // 64)
    end_row, end_col = rc_from_square(index % 64)
    return start_row, start_col, end_row, end_col

def legal_move_indices(board: chess.Board) -> dict:
    """
    Maps each flat index to its legal python-chess move, promoting to a queen when the
    same from/to squares allow several promotion pieces.
    """
    moves = {}
    for move in board.legal_moves:
        if move.promotion not in (None, chess.QUEEN):
            continue
        moves[encode_chess_move(move)] = move
    return moves

__all__ = [
    "PIECE_PLANES", "encode_board", "encode_move", "encode_chess_move",
    "decode_move_index", "legal_move_indices", "square_from_rc", "rc_from_square",
]

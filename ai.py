import chess
import os
from typing import Union, List
if "COLAB_GPU" in os.environ:
    BASE_DIR = "/content/drive/MyDrive/KnightVision"
else:
    BASE_DIR = os.environ.get("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

PIECE_TO_INDEX = {
    "wK": 0, "wQ": 1, "wR": 2, "wB": 3, "wN": 4, "wp": 5,
    "bK": 6, "bQ": 7, "bR": 8, "bB": 9, "bN": 10, "bp": 11
}

INDEX_TO_PIECE = { v:k for k, v in PIECE_TO_INDEX.items()}

# Precompute row, col for each square
SQUARE_TO_RC = {sq: (7 - sq // 8, sq % 8) for sq in chess.SQUARES}

def encode_board(board: Union[chess.Board, List[List[str]], np.ndarray]) -> np.ndarray:
    # Initialize empty board tensor
    encoded = np.zeros((12, 8, 8), dtype=np.float32)

    # If a nested list or ndarray is passed, convert to python-chess Board
    if isinstance(board, (list, np.ndarray)):
        # Expect board representation as codes matching PIECE_TO_INDEX
        arr = np.array(board)
        for row, col in zip(*np.nonzero(arr)):
            key = arr[row, col]
            idx = PIECE_TO_INDEX.get(key)
            if idx is not None:
                encoded[idx, row, col] = 1.0
        return encoded

    # Otherwise assume a python-chess Board
    for square, piece in board.piece_map().items():
        idx_color = 'w' if piece.color == chess.WHITE else 'b'
        letter = 'p' if piece.piece_type == chess.PAWN else piece.symbol().upper()
        key = f"{idx_color}{letter}"
        idx = PIECE_TO_INDEX[key]
        row, col = SQUARE_TO_RC[square]
        encoded[idx, row, col] = 1.0

    return encoded

def decode_move_index(index):
    """
    Converts a flat index (0-4095) back to a move (start_row, start_col, end_row, end_col).
    """
    start = index // 64
    end = index % 64
    return(start // 8, start % 8 , end // 8, end % 8)

def encode_move(start_row, start_col, end_row, end_col):
    """
    Converts a move into a flat index (0-4095) for a classifier output.
    """
    start = start_row * 8 + start_col
    end = end_row * 8 + end_col
    return start * 64 + end

__all__ = ["encode_board", "decode_move_index", "encode_move"]
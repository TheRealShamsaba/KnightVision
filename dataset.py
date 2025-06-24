import os
import logging
from logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/KnightVision"
except ImportError:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

logger.info("[DATASET] Base directory set to: %s", BASE_DIR)

import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import chess
import chess.pgn
import numpy as np
import random

PIECE_TO_IDX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

class ChessDataset(Dataset):
    def __init__(self, jsonl_path, move_to_idx=None, max_games=None):
        self.data = []
        self.move_to_idx = move_to_idx or {}
        self.idx_to_move = {}  # for decoding if needed
        jsonl_path = os.path.join(BASE_DIR, jsonl_path) if not os.path.isabs(jsonl_path) else jsonl_path
        try:
            with open(jsonl_path, 'r') as f:
                for i, line in enumerate(f):
                    if max_games and i >= max_games:
                        break
                    game = json.loads(line)
                    fen, move, outcome = game['fen'], game['move'], game.get('outcome')

                    if move not in self.move_to_idx:
                        idx = len(self.move_to_idx)
                        self.move_to_idx[move] = idx
                        self.idx_to_move[idx] = move

                    board_tensor = self.fen_to_tensor(fen)
                    move_idx = self.move_to_idx[move]
                    self.data.append((board_tensor, move_idx))
            logger.info("[DATASET] Loaded %s samples from %s", len(self.data), jsonl_path)
            logger.info("[DATASET] Unique moves encoded: %s", len(self.move_to_idx))
        except Exception as e:
            logger.error("[ERROR] Failed to load dataset from %s: %s", jsonl_path, e)

    def fen_to_tensor(self, fen):
        board = chess.Board(fen)
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = PIECE_TO_IDX[piece.symbol()]
                row, col = divmod(square, 8)
                tensor[idx][7 - row][col] = 1
        return torch.tensor(tensor)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def extend(self, games):
        """
        Extend the dataset with additional self-play games.
        Each game should be a dict with 'fen' and 'move' keys.
        """
        for game in games:
            fen = game['fen']
            move = game['move']
            if move not in self.move_to_idx:
                idx = len(self.move_to_idx)
                self.move_to_idx[move] = idx
                self.idx_to_move[idx] = move
            board_tensor = self.fen_to_tensor(fen)
            move_idx = self.move_to_idx[move]
            self.data.append((board_tensor, move_idx))


def create_dataloaders(
    jsonl_path, batch_size=64, val_split=0.1, max_games=None,
    num_workers=os.cpu_count(), pin_memory=torch.cuda.is_available(),
    seed: int = 42
):
    dataset = ChessDataset(jsonl_path, max_games=max_games)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, dataset.move_to_idx

# Alias for compatibility
ChessPGNDataset = ChessDataset
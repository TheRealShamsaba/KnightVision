import os
import sys
import multiprocessing as mp
import json
import time
import gc
import logging
import traceback
import zipfile
import psutil
import numpy as np
import torch
from datetime import datetime
from types import SimpleNamespace
from model_utils import load_or_initialize_model
# Add logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from dotenv import load_dotenv
load_dotenv()

import random
import requests

from telegram_utils import send_telegram_message
from train import train_with_validation as train_model, ChessPGNDataset
from torch.utils.data import DataLoader
from model import ChessNet
import torch.optim as optim
from stockfish_play import play_vs_stockfish
from self_play import generate_self_play_data

# --- Reproducibility: fixed global seed ---
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Enforce deterministic behavior in cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


print("✅ Script loaded.", flush=True)
sys.stdout.flush()
sys.stderr.flush()

# --- Helper to escape unsafe Markdown for Telegram
def safe_send_telegram(msg):
    if not msg.strip():
        print("⚠️ Skipping empty Telegram message.")
        return
    try:
        safe_msg = msg.replace("*", "\\*").replace("_", "\\_").replace("[", "\\[").replace("]", "\\]")
        send_telegram_message(safe_msg)
    except Exception as e:
        print("⚠️ Telegram failed:", e)

def format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

# === Unified configuration dataclass or namespace ===
def build_cfg():
    # Paths and hyperparameters from env or defaults
    base_dir = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    data_dir = os.getenv("DATA_DIR", os.path.join(base_dir, "data"))
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", os.path.join(base_dir, "checkpoints"))
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    games_path = os.getenv("GAMES_PATH", os.path.join(data_dir, "games.jsonl"))
    patience = int(os.getenv("PATIENCE", "3"))
    num_selfplay_games = int(os.getenv("NUM_SELFPLAY_GAMES", "5"))
    selfplay_max_moves = os.getenv("SELFPLAY_MAX_MOVES")
    selfplay_max_moves = int(selfplay_max_moves) if selfplay_max_moves is not None else None
    train_epochs = int(os.getenv("TRAIN_EPOCHS", "2"))
    batch_size = int(os.getenv("BATCH_SIZE", "2048"))
    lr = float(os.getenv("LR", "1e-3"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stockfish_cfg = SimpleNamespace(path=os.getenv("STOCKFISH_PATH", "stockfish"), depth=int(os.getenv("STOCKFISH_DEPTH", "10")))
    # Logging configuration values before returning
    logger.info("Configuration:")
    for k, v in locals().items():
        if k != 'self':
            logger.info("  %s = %s", k, v)
    return SimpleNamespace(
        base_dir=base_dir,
        data_dir=data_dir,
        checkpoint_dir=checkpoint_dir,
        games_path=games_path,
        patience=patience,
        selfplay=SimpleNamespace(
            num_games=num_selfplay_games,
            max_moves=selfplay_max_moves
        ),
        train=SimpleNamespace(
            epochs=train_epochs,
            batch_size=batch_size,
            lr=lr
        ),
        device=device,
        checkpoint_path=os.path.join(checkpoint_dir, "model_latest.pth"),
        stockfish=stockfish_cfg
    )

def reinforcement_loop(cfg):
    # Stage 1: load or initialize model
    model, optimizer, start_epoch = _load_model_helper(
        ChessNet,
        optim.Adam,
        {'lr': cfg.train.lr},
        cfg.checkpoint_path,
        cfg.device
    )
    # Stage 2: prepare dataset and split
    from torch.utils.data import random_split
    dataset = ChessPGNDataset(cfg.games_path)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    # Stage 3: train with validation
    train_with_validation(model, cfg, train_dataset, val_dataset)

    # Stage 4: generate self-play data
    new_games = generate_self_play_data(model, cfg.selfplay.num_games, cfg.device, cfg.selfplay.max_moves)
    logger.info("♟️ Self-play generated %d games", len(new_games))
    safe_send_telegram(f"♟️ Self-play generated {len(new_games)} games")

    # Stage 5: extend dataset with new self-play games
    if new_games:
        dataset.extend(new_games)

    # Stage 6: evaluate vs Stockfish
    play_vs_stockfish(model, cfg.stockfish)


# --- Helper for new reinforcement_loop ---
def _load_model_helper(
    model_class,
    optimizer_class,
    optimizer_kwargs,
    checkpoint_path,
    device
):
    logger.info(f"🔄 Loading model from {checkpoint_path}…")
    model, optimizer, start_epoch = load_or_initialize_model(
        model_class=model_class,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        checkpoint_path=checkpoint_path,
        device=device
    )
    model.to(device)
    return model, optimizer, start_epoch

from torch.utils.data import random_split

def train_with_validation(model, cfg, train_dataset=None, val_dataset=None):
    from train import ChessPGNDataset
    import torch.optim as optim
    # Prepare dataset if not provided
    if train_dataset is None or val_dataset is None:
        dataset = ChessPGNDataset(cfg.games_path)
        n_total = len(dataset)
        n_train = int(0.9 * n_total)
        n_val = n_total - n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    # Call train_model and return its result
    return train_model(
        model,
        train_dataset,
        val_dataset,
        optimizer,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        device=cfg.device,
        checkpoint_path=cfg.checkpoint_path
    )

def main():
    mp.set_start_method("spawn", force=True)
    cfg = build_cfg()
    # Optionally: prepare datasets here and pass to reinforcement_loop if needed
    reinforcement_loop(cfg)

if __name__ == "__main__":
    main()

# --- Check for already running instance ---
def check_if_already_running():
    current_pid = os.getpid()
    script_name = os.path.basename(__file__)
    count = 0
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] != current_pid and script_name in ' '.join(proc.info['cmdline']):
                count += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    if count > 0:
        print("⚠️ Another instance of this script is already running! Exiting to avoid duplication.")
        sys.exit(1)
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
from dotenv import load_dotenv
import random
import requests

from telegram_utils import send_telegram_message
from train import train as train_model, ChessPGNDataset
from torch.utils.data import DataLoader, random_split
from model import ChessNet
import torch.optim as optim
from stockfish_play import play_vs_stockfish
from self_play import generate_self_play_data

# --- Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()

# --- Reproducibility ---
SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("✅ Script loaded.", flush=True)
sys.stdout.flush()
sys.stderr.flush()

# --- Helper for safe Telegram messages ---
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

# --- Session directory logic (copied from train.py) ---
BASE_SESSIONS_DIR = "/content/drive/MyDrive/KnightVision/sessions"
os.makedirs(BASE_SESSIONS_DIR, exist_ok=True)

def create_new_session_dir():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(BASE_SESSIONS_DIR, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir

def find_last_session_dir():
    runs = [os.path.join(BASE_SESSIONS_DIR, d) for d in os.listdir(BASE_SESSIONS_DIR)]
    runs = [d for d in runs if os.path.isdir(d)]
    if not runs:
        return None
    runs.sort(key=os.path.getmtime, reverse=True)
    return runs[0]

# --- Build configuration ---
def build_cfg(session_dir):
    checkpoint_dir = os.path.join(session_dir, "checkpoints")
    data_dir = "/content/drive/MyDrive/KnightVision/data"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    games_path = os.path.join(data_dir, "games.jsonl")
    patience = int(os.getenv("PATIENCE", "3"))
    num_selfplay_games = int(os.getenv("NUM_SELFPLAY_GAMES", "5"))
    selfplay_max_moves = os.getenv("SELFPLAY_MAX_MOVES")
    selfplay_max_moves = int(selfplay_max_moves) if selfplay_max_moves is not None else None
    train_epochs = int(os.getenv("TRAIN_EPOCHS", "2"))
    batch_size = int(os.getenv("BATCH_SIZE", "2048"))
    lr = float(os.getenv("LR", "1e-3"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch_LAST.pth")
    model_path = last_checkpoint_path if os.path.exists(last_checkpoint_path) else best_checkpoint_path

    stockfish_cfg = SimpleNamespace(
        path=os.getenv("STOCKFISH_PATH", "/usr/bin/stockfish"),
        depth=int(os.getenv("STOCKFISH_DEPTH", "10"))
    )

    logger.info("Configuration:")
    for k, v in locals().items():
        if k in {"stockfish_cfg"}:
            continue
        logger.info("  %s = %s", k, v)
    # Then print stockfish separately
    logger.info(f"  Stockfish path: {stockfish_cfg.path}")
    logger.info(f"  Stockfish depth: {stockfish_cfg.depth}")

    return SimpleNamespace(
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
        model_path=model_path,
        stockfish=stockfish_cfg
    )

# --- Reinforcement loop ---
def reinforcement_loop(cfg):
    model, optimizer, start_epoch = _load_model_helper(
        ChessNet,
        optim.Adam,
        {'lr': cfg.train.lr},
        cfg.model_path,
        cfg.device
    )

    dataset = ChessPGNDataset(cfg.games_path)
    n_total = len(dataset)
    n_train = int(0.9 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    num_iterations = int(os.getenv("NUM_ITERATIONS", "5"))

    for iteration in range(1, num_iterations + 1):
        logger.info(f"=== Iteration {iteration}/{num_iterations} ===")

        # Train on current data
        train_model(
            model,
            optimizer,
            start_epoch,
            train_dataset,
            val_dataset,
            epochs=cfg.train.epochs,
            batch_size=cfg.train.batch_size,
            device=cfg.device
        )
        safe_send_telegram(f"✅ Iteration {iteration}: Training completed")

        # Generate self-play games
        new_games = generate_self_play_data(
            model,
            cfg.selfplay.num_games,
            cfg.device,
            cfg.selfplay.max_moves
        )
        logger.info("♟️ Self-play generated %d games", len(new_games))
        safe_send_telegram(f"♟️ Iteration {iteration}: Self-play generated {len(new_games)} games")

        # Extend dataset
        if new_games:
            dataset.extend(new_games)
            # Update splits
            n_total = len(dataset)
            n_train = int(0.9 * n_total)
            n_val = n_total - n_train
            train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

        # Optionally evaluate
        if cfg.selfplay.num_games > 0:
            play_vs_stockfish(model, cfg.selfplay.num_games, cfg.stockfish.path)

    logger.info("🏁 All iterations completed!")
    safe_send_telegram("🏁 All reinforcement iterations completed!")

# --- Helper for model loading ---
def _load_model_helper(model_class, optimizer_class, optimizer_kwargs, model_path, device):
    logger.info(f"🔄 Loading model from {model_path}…")
    model, optimizer, start_epoch = load_or_initialize_model(
        model_class=model_class,
        optimizer_class=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        model_path=model_path,
        device=device
    )
    model.to(device)
    return model, optimizer, start_epoch

# --- Train wrapper ---
def train_with_validation(model, cfg, train_dataset=None, val_dataset=None):
    if train_dataset is None or val_dataset is None:
        dataset = ChessPGNDataset(cfg.games_path)
        n_total = len(dataset)
        n_train = int(0.9 * n_total)
        n_val = n_total - n_train
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)
    return train_model(
        model,
        train_dataset,
        val_dataset,
        optimizer,
        epochs=cfg.train.epochs,
        batch_size=cfg.train.batch_size,
        device=cfg.device
    )

# --- Main ---
def main():
    check_if_already_running()
    mp.set_start_method("spawn", force=True)

    if os.getenv("RESUME_LAST_SESSION", "False") == "True":
        last_session_dir = find_last_session_dir()
        if last_session_dir is None:
            session_dir = create_new_session_dir()
            logger.info("No existing session found. Created new session.")
        else:
            session_dir = last_session_dir
            logger.info(f"Resuming session: {session_dir}")
    else:
        session_dir = create_new_session_dir()
        logger.info(f"Starting new session: {session_dir}")

    cfg = build_cfg(session_dir)
    reinforcement_loop(cfg)

if __name__ == "__main__":
    main()
from typing import List, Tuple, Any
import torch.nn as nn
import os
import time
import numpy as np
import multiprocessing as mp
import glob
# Force 'fork' start method for multiprocessing (important for PyTorch and avoiding heavy import re-execution)
if mp.get_start_method(allow_none=True) != 'fork':
    mp.set_start_method('fork', force=True)
# Dirichlet noise parameters
EPSILON = float(os.getenv("DIR_NOISE_EPS", "0.25"))
ALPHA = float(os.getenv("DIR_NOISE_ALPHA", "0.3"))
import logging
from ai.logging_utils import configure_logging
# initialize module-level logger for both main and worker processes
configure_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)
from ai import encode_board
import datetime
import torch
import random, numpy as np, os, torch

SEED = int(os.getenv("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# global device for self-play
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = int(os.getenv("SELFPLAY_BATCH_SIZE", "16"))

# --- reproducibility / seeding for self-play ---
SEED = int(os.getenv("SEED", "42"))
import random as _random
_random.seed(SEED)
import numpy as _np
_np.random.seed(SEED)
import torch as _torch
_torch.manual_seed(SEED)
if _torch.cuda.is_available():
    _torch.cuda.manual_seed_all(SEED)
# make CUDNN deterministic
_torch.backends.cudnn.deterministic = True
_torch.backends.cudnn.benchmark = False

# -- process pool initializer for shared model loading --
_shared_model = None
def _init_worker(model_path, device_str, seed):
    import os, random, numpy as np, torch, glob
    from ai.model import ChessNet
    global _shared_model, device

    # set device
    device = torch.device(device_str)
    # Logging initialization for worker
    import logging
    from ai.logging_utils import configure_logging
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    global logger
    logger = logging.getLogger(__name__)

    # Check if model_path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Specified model checkpoint not found: {model_path}")

    # load and prepare model
    m = ChessNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        m.load_state_dict(checkpoint["model_state_dict"])
    else:
        m.load_state_dict(checkpoint)
    m.eval()
    _shared_model = m

    # reseed RNGs for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Removed optimizer initialization to avoid starting a new optimizer


from core.chessEngine import GameState
from ai import encode_move
import random
import torch
import logging
from ai.model import ChessNet
import psutil

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("🖥️ Using device: %s", device)

    if torch.cuda.is_available():
        logger.info("💾 VRAM used: %.2f MB", torch.cuda.memory_allocated(device) / 1024 ** 2)

    # logger = logging.getLogger(__name__)  # Already defined at module level

    model = ChessNet().to(device)
    model.eval()


def _run_single_game(game_idx, sleep_time, max_moves=80):
    global _shared_model, device, logger
    model = _shared_model
    logger.info("🕹️ Starting game %s", game_idx + 1)
    logger.debug("⏳ Game initialization complete — entering move loop")
    gs = GameState()
    game_data = []
    move_count = 0
    maxed_out = False
    buffer = []

    while True:  # Continue until the game ends naturally or max moves reached
        valid_moves = gs.getValidMoves()
        logger.debug("♟️ Valid moves count: %s", len(valid_moves))
        if not valid_moves:
            break

        # Accumulate for batched inference
        buffer.append(encode_board(gs.board))
        # When buffer full or last move, run batch
        if len(buffer) >= BATCH_SIZE:
            with torch.no_grad():
                batch_np = np.stack(buffer, axis=0).astype(np.float32)
                batch_tensor = torch.from_numpy(batch_np).to(device)
                batch_policy, batch_value = model(batch_tensor)
                _run_single_game._last_outputs = (batch_policy.cpu().numpy(), batch_value.cpu().numpy())
                buffer.clear()
        # Ensure we have inference outputs before retrieving
        if not hasattr(_run_single_game, "_last_outputs"):
            with torch.no_grad():
                batch_np = np.stack(buffer, axis=0).astype(np.float32)
                batch_tensor = torch.from_numpy(batch_np).to(device)
                batch_policy, batch_value = _shared_model(batch_tensor)
                _run_single_game._last_outputs = (batch_policy.cpu().numpy(), batch_value.cpu().numpy())
                buffer.clear()
        # Retrieve last output
        policy_logits = torch.from_numpy(_run_single_game._last_outputs[0][-1]).unsqueeze(0)
        value_logits = torch.from_numpy(_run_single_game._last_outputs[1][-1]).unsqueeze(0)

        policy = torch.softmax(policy_logits.squeeze(), dim=0).detach().cpu().numpy()

        # Add Dirichlet exploration noise
        noise = np.random.dirichlet([ALPHA] * policy.shape[0])
        policy = (1 - EPSILON) * policy + EPSILON * noise
        # Log policy distribution stats
        entropy = -np.sum(policy * np.log(policy + 1e-8))
        logger.debug("Policy entropy: %.4f (mean prob: %.4f)", entropy, policy.mean())

        legal_indices = [encode_move(m.startRow, m.startCol, m.endRow, m.endCol) for m in valid_moves]
        legal_probs = [policy[i] if i < len(policy) else 0 for i in legal_indices]

        total_weight = sum(legal_probs)
        if total_weight == 0:
            move = random.choice(valid_moves)
        else:
            normalized = [w / total_weight for w in legal_probs]
            move = random.choices(valid_moves, weights=normalized, k=1)[0]

        # Optionally log top-3 move probabilities
        top_idxs = np.argsort(policy)[-3:][::-1]
        logger.debug("Top moves indices: %s with probs %s", top_idxs.tolist(), policy[top_idxs].tolist())

        move_index = encode_move(move.startRow, move.startCol, move.endRow, move.endCol)
        game_data.append((encode_board(gs.board), move_index))
        if len(game_data) % 10 == 0:
            logger.debug("♟️ Played %s moves so far...", len(game_data))
        gs.makeMove(move)
        move_count += 1

        if gs.isDraw():
            logger.info("⚠️ Draw detected early; ending game.")
            break

        # Resign condition: if predicted value < threshold after min_moves
        if move_count > 15 and value_logits.item() < -0.7:
            logger.warning("⚠️ Resignation triggered by low value prediction.")
            outcome = -1 if gs.whiteToMove else 1
            result_reason = "Resignation"
            break

        # # (Future: Batched inference, board symmetries, etc. can be inserted here)

        # TODO: Dynamically adjust max_moves as model strength improves

        # enforce move limit (only if a max_moves value was provided)
        if max_moves is not None and move_count >= max_moves:
            logger.warning(f"⚠️ Max moves reached ({max_moves}); terminating game early.")
            maxed_out = True
            break

    # Flush any remaining batched boards for final moves
    if buffer:
        with torch.no_grad():
            batch_np = np.stack(buffer, axis=0).astype(np.float32)
            batch_tensor = torch.from_numpy(batch_np).to(device)
            batch_policy, batch_value = _shared_model(batch_tensor)
            _run_single_game._last_outputs = (batch_policy.cpu().numpy(), batch_value.cpu().numpy())
            buffer.clear()

    # Determine outcome, with max-move override
    if maxed_out:
        outcome = 0  # Treat as draw
        result_reason = f"Max moves ({max_moves}) reached"
    elif 'outcome' in locals() and 'result_reason' in locals():
        # resignation outcome already set
        pass
    elif gs.inCheck() and len(gs.getValidMoves()) == 0:
        # Checkmate
        outcome = 1 if not gs.whiteToMove else -1
        result_reason = "Checkmate"
    elif len(gs.getValidMoves()) == 0:
        # Stalemate
        outcome = 0
        result_reason = "Stalemate"
    elif gs.isDraw():
        # Draw by repetition or 50-move rule
        outcome = 0
        result_reason = "Draw (50-move or repetition)"
    else:
        white_material = sum(piece_value(p) for r in gs.board for p in r if p.isupper())
        black_material = sum(piece_value(p) for r in gs.board for p in r if p.islower())
        if white_material > black_material:
            outcome = 1 if not gs.whiteToMove else -1
        elif black_material > white_material:
            outcome = -1 if not gs.whiteToMove else 1
        else:
            outcome = 0
        result_reason = "Material-based final evaluation"
    logger.info("✅ Game %s complete. Moves played: %s | Outcome: %s (%s)", game_idx + 1, len(game_data), outcome, result_reason)
    logger.debug("🧠 RAM usage: %s%%", psutil.virtual_memory().percent)
    if torch.cuda.is_available():
        logger.debug("💾 VRAM: %.2f MB", torch.cuda.memory_allocated(device) / 1024 ** 2)

    # Adjusted reward logic
    if outcome == 1:   # Win
        reward = 1.0
    elif outcome == 0: # Draw
        reward = 0.2
    else:              # Loss
        reward = -1.0

    # Attach reward (not just outcome) to each move data
    game_data_with_outcome = [(state, move, reward) for (state, move) in game_data]

    return game_idx, game_data_with_outcome


def self_play(model, num_games, device, max_moves=None, model_path=None):
    if model is None and model_path is None:
        raise ValueError("Either a model instance or model_path must be provided.")
    logger.info("Starting self-play with %s games...", num_games)
    data = []
    SEQUENTIAL = os.getenv("SELFPLAY_SEQ", "0") == "1"
    WORKERS = int(os.getenv("SELFPLAY_WORKERS", str(min(num_games, os.cpu_count() or 1))))
    global _shared_model
    if model_path is not None:
        # Load model from checkpoint path
        if SEQUENTIAL or WORKERS <= 1:
            logger.info("🔁 Running self-play sequentially with %s games", num_games)
            _init_worker(model_path, device.type, SEED)
            results = [_run_single_game(idx, 0.0, max_moves) for idx in range(num_games)]
        else:
            logger.info("🔁 Running self-play in parallel with %s workers", WORKERS)
            pool = mp.Pool(
                processes=WORKERS,
                initializer=_init_worker,
                initargs=(model_path, device.type, SEED)
            )
            tasks = [(idx, 0.0, max_moves) for idx in range(num_games)]
            results = pool.starmap(_run_single_game, tasks)
            pool.close()
            pool.join()
    else:
        # Use the provided model instance directly
        _shared_model = model
        logger.info("🔁 Running self-play sequentially with %s games (model instance provided)", num_games)
        results = [_run_single_game(idx, 0.0, max_moves) for idx in range(num_games)]
    for idx, game_data in results:
        data.extend(game_data)
    logger.info("✅ Completed all self-play games: %s/%s", num_games, num_games)
    return data


# Piece value function for material evaluation
def piece_value(piece):
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    return values.get(piece.upper(), 0)


def generate_self_play_data(model: nn.Module, num_games: int, device: torch.device, max_moves: int = None) -> List[Tuple[Any, int, float]]:
    # The self_play function now handles whether to use a model instance or model_path.
    data = self_play(model, num_games, device, max_moves)
    # Filter for decisive games only (win or loss)
    decisive_data = [record for record in data if record[2] == 1.0 or record[2] == -1.0]
    MIN_DECISIVE_GAMES = 10
    if len(decisive_data) < MIN_DECISIVE_GAMES:
        print(f"⚠️ Only {len(decisive_data)} decisive games generated; consider generating more or adjusting parameters.")
    else:
        print(f"✅ Using {len(decisive_data)} decisive self-play games for training.")
        data = decisive_data
    return data

from typing import List, Tuple, Any
import torch.nn as nn
import os
import time
import numpy as np
import multiprocessing as mp
import glob
# Ensure BASE_DIR is defined for all module contexts
BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Force 'fork' start method for multiprocessing (important for PyTorch and avoiding heavy import re-execution)
if mp.get_start_method(allow_none=True) != 'fork':
    mp.set_start_method('fork', force=True)
# Dirichlet noise parameters
EPSILON = float(os.getenv("DIR_NOISE_EPS", "0.25"))
ALPHA = float(os.getenv("DIR_NOISE_ALPHA", "0.3"))
import logging
from logging_utils import configure_logging
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
    from model import ChessNet
    global _shared_model, device

    # set device
    device = torch.device(device_str)
    # Logging initialization for worker
    import logging
    from logging_utils import configure_logging
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    global logger
    logger = logging.getLogger(__name__)

    # Auto-discover checkpoint if original path missing
    if not os.path.exists(model_path):
        base = os.getenv("BASE_DIR", BASE_DIR)
        checkpoints_dir = os.path.join(base, "runs", "chess_rl_v2", "checkpoints")
        found = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
        if found:
            model_path = max(found, key=os.path.getmtime)
            print(f"⚠️ _init_worker: auto-selected checkpoint '{model_path}'")
        else:
            fallback = os.path.join(base, "checkpoints", "model.pth")
            if os.path.exists(fallback):
                model_path = fallback
                print(f"⚠️ _init_worker: using fallback checkpoint '{model_path}'")
            else:
                try:
                    contents = os.listdir(checkpoints_dir)
                except Exception:
                    contents = []
                raise FileNotFoundError(
                    f"❌ No model found!\n"
                    f"  Tried original path: {model_path}\n"
                    f"  Tried fallback     : {fallback}\n"
                    f"  {checkpoints_dir} contains: {contents}"
                )

    # load and prepare model
    m = ChessNet().to(device)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    _shared_model = m

    # reseed RNGs for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


from chessEngine import GameState
from ai import encode_move
import random
import torch
import logging
from model import ChessNet
import psutil

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("🖥️ Using device: %s", device)

    if torch.cuda.is_available():
        logger.info("💾 VRAM used: %.2f MB", torch.cuda.memory_allocated(device) / 1024 ** 2)

    # logger = logging.getLogger(__name__)  # Already defined at module level

    model = ChessNet().to(device)
    model.eval()


def _run_single_game(game_idx, sleep_time, max_moves):
    global _shared_model, device, logger
    model = _shared_model
    logger.info("🕹️ Starting game %s", game_idx + 1)
    logger.debug("⏳ Game initialization complete — entering move loop")
    gs = GameState()
    game_data = []
    move_count = 0
    maxed_out = False

    while True:  # Continue until the game ends naturally or max moves reached
        valid_moves = gs.getValidMoves()
        logger.debug("♟️ Valid moves count: %s", len(valid_moves))
        if not valid_moves:
            break

        # Accumulate for batched inference
        if not hasattr(_run_single_game, "_buffer"):
            _run_single_game._buffer = []
        _run_single_game._buffer.append(encode_board(gs.board))
        # When buffer full or last move, run batch
        if len(_run_single_game._buffer) >= BATCH_SIZE:
            batch_np = np.stack(_run_single_game._buffer, axis=0).astype(np.float32)
            batch_tensor = torch.from_numpy(batch_np).to(device)
            with torch.no_grad():
                batch_policy, batch_value = model(batch_tensor)
            # Store back for each entry
            _run_single_game._last_outputs = (batch_policy.cpu().numpy(), batch_value.cpu().numpy())
            _run_single_game._buffer.clear()
        # Ensure we have inference outputs before retrieving
        if not hasattr(_run_single_game, "_last_outputs"):
            batch_np = np.stack(_run_single_game._buffer, axis=0).astype(np.float32)
            batch_tensor = torch.from_numpy(batch_np).to(device)
            with torch.no_grad():
                batch_policy, batch_value = _shared_model(batch_tensor)
            _run_single_game._last_outputs = (batch_policy.cpu().numpy(), batch_value.cpu().numpy())
            _run_single_game._buffer.clear()
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

        # # (Future: Batched inference, board symmetries, etc. can be inserted here)

        # enforce move limit (only if a max_moves value was provided)
        if max_moves is not None and move_count >= max_moves:
            logger.warning(f"⚠️ Max moves reached ({max_moves}); terminating game early.")
            maxed_out = True
            break

        if sleep_time:
            time.sleep(sleep_time)

    # Flush any remaining batched boards for final moves
    if hasattr(_run_single_game, "_buffer") and _run_single_game._buffer:
        batch_np = np.stack(_run_single_game._buffer, axis=0).astype(np.float32)
        batch_tensor = torch.from_numpy(batch_np).to(device)
        with torch.no_grad():
            batch_policy, batch_value = _shared_model(batch_tensor)
        _run_single_game._last_outputs = (batch_policy.cpu().numpy(), batch_value.cpu().numpy())
        _run_single_game._buffer.clear()

    # Determine outcome, with max-move override
    if maxed_out:
        outcome = 0.5
        result_reason = f"Max moves ({max_moves}) reached"
    elif gs.inCheck() and len(gs.getValidMoves()) == 0:
        # Checkmate
        outcome = 1 if not gs.whiteToMove else -1
        result_reason = "Checkmate"
    elif len(gs.getValidMoves()) == 0:
        # Stalemate
        outcome = 0.5
        result_reason = "Stalemate"
    elif gs.isDraw():
        # Draw by repetition or 50-move rule
        outcome = 0.5
        result_reason = "Draw (50-move or repetition)"
    else:
        # Material-based score for early termination
        white_material = sum(piece_value(p) for r in gs.board for p in r if p.isupper())
        black_material = sum(piece_value(p) for r in gs.board for p in r if p.islower())
        if white_material == black_material:
            outcome = 0
        else:
            outcome = (white_material - black_material) / max(white_material, black_material)
        result_reason = "Material difference"
    logger.info("✅ Game %s complete. Moves played: %s | Outcome: %s", game_idx + 1, len(game_data), outcome)
    logger.debug("🧠 RAM usage: %s%%", psutil.virtual_memory().percent)
    if torch.cuda.is_available():
        logger.debug("💾 VRAM: %.2f MB", torch.cuda.memory_allocated(device) / 1024 ** 2)

    # Attach outcome to each move data
    game_data_with_outcome = [(state, move, outcome) for (state, move) in game_data]

    return game_idx, game_data_with_outcome


def self_play(model, num_games, device, max_moves=None):
    logger.info("Starting self-play with %s games...", num_games)
    data = []
    SEQUENTIAL = os.getenv("SELFPLAY_SEQ", "0") == "1"
    WORKERS = int(os.getenv("SELFPLAY_WORKERS", str(min(num_games, os.cpu_count() or 1))))
    candidate = os.path.join(BASE_DIR, "runs", "chess_rl_v2", "checkpoints", "model_latest.pth")
    if os.path.exists(candidate):
        model_path = candidate
    else:
        model_path = os.path.join(BASE_DIR, "checkpoints", "model.pth")
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
    for idx, game_data in results:
        data.extend(game_data)
    logger.info("✅ Completed all self-play games: %s/%s", num_games, num_games)
    return data


# Piece value function for material evaluation
def piece_value(piece):
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    return values.get(piece.upper(), 0)


def generate_self_play_data(model: nn.Module, num_games: int, device: torch.device, max_moves: int = None) -> List[Tuple[Any, int, float]]:
    global _shared_model
    _shared_model = model
    return self_play(model, num_games, device, max_moves)


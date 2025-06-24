# self_play.py
import os
import time
import numpy as np
import multiprocessing as mp
# Dirichlet noise parameters
EPSILON = float(os.getenv("DIR_NOISE_EPS", "0.25"))
ALPHA = float(os.getenv("DIR_NOISE_ALPHA", "0.3"))
from telegram_utils import send_telegram_message
import logging
from logging_utils import configure_logging
from ai import encode_board
import datetime
import tensorflow as tf
import torch

BATCH_SIZE = int(os.getenv("SELFPLAY_BATCH_SIZE", "16"))

# --- reproducibility / seeding for self-play ---
SEED = int(os.getenv("SEED", "42"))
import random as _random
_random.seed(SEED)
import numpy as _np
_np.random.seed(SEED)
import tensorflow as _tf
_tf.random.set_seed(SEED)
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
    import os, random, numpy as np, torch
    from model import ChessNet
    global _shared_model, device
    # device setup
    device = torch.device(device_str)
    # load model
    m = ChessNet().to(device)
    m.load_state_dict(torch.load(model_path, map_location=device))
    m.eval()
    _shared_model = m
    # reseed RNGs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

# Ensure BASE_DIR is defined before use
BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# set up TensorFlow log directory for self-play
SELFPLAY_LOG_DIR = os.path.join(BASE_DIR, "runs", "self_play", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(SELFPLAY_LOG_DIR, exist_ok=True)
tf_writer = tf.summary.create_file_writer(SELFPLAY_LOG_DIR)
configure_logging()
logger = logging.getLogger(__name__)
logger.info("Self-play script loaded...")
logger.info("✅ Telegram test message dispatched.")
logger.info("📋 Note: All Telegram messages will log their intent before sending.")

try:
    send_telegram_message("📥 self_play.py loaded successfully.")
except Exception as e:
    logger.error("⚠️ Telegram send failed: %s", e)

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=True)
        BASE_DIR = "/content/drive/MyDrive/KnightVision"
    except Exception as e:
        logger.error("⚠️ Colab drive mount failed: %s", e)
        BASE_DIR = os.getenv("BASE_DIR", "/content/drive/MyDrive/KnightVision")
else:
    logger.info("📦 Not running in Colab — skipping drive.mount")
    from dotenv import load_dotenv
    load_dotenv()
    BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chessEngine import GameState
from ai import encode_move
import random
import torch
import logging
from model import ChessNet
import psutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("🖥️ Using device: %s", device)

if torch.cuda.is_available():
    logger.info("💾 VRAM used: %.2f MB", torch.cuda.memory_allocated(device) / 1024 ** 2)

logger = logging.getLogger(__name__)

model = ChessNet().to(device)
model.eval()


def _run_single_game(game_idx, sleep_time, max_moves):
    global _shared_model, device
    model = _shared_model
    logger.info("🕹️ Starting game %s/%s", game_idx + 1, "N/A")
    # Telegram notification for each game start
    try:
        send_telegram_message(f"🕹️ Game {game_idx+1}/N/A starting now.")
    except Exception:
        logger.error("⚠️ Telegram send failed for game start.")
    # TensorFlow log for game start
    with tf_writer.as_default():
        tf.summary.scalar("SelfPlay/GameStart", 1, step=int(game_idx + 1))
    tf_writer.flush()
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
        # Retrieve last output
        policy_logits = torch.from_numpy(_run_single_game._last_outputs[0][-(1)]).unsqueeze(0)
        value_logits = torch.from_numpy(_run_single_game._last_outputs[1][-(1)]).unsqueeze(0)

        policy = torch.softmax(policy_logits.squeeze(), dim=0).detach().cpu().numpy()

        # Add TensorBoard histogram for policy distribution
        with tf_writer.as_default():
            tf.summary.histogram("SelfPlay/PolicyDist", policy, step=int(game_idx + 1))
        tf_writer.flush()

        # Add Dirichlet exploration noise
        noise = np.random.dirichlet([ALPHA] * policy.shape[0])
        policy = (1 - EPSILON) * policy + EPSILON * noise
        # Log policy distribution stats
        entropy = -np.sum(policy * np.log(policy + 1e-8))
        logger.debug("Policy entropy: %.4f (mean prob: %.4f)", entropy, policy.mean())
        with tf_writer.as_default():
            tf.summary.scalar("SelfPlay/PolicyEntropy", float(entropy), step=int(game_idx + 1))
        tf_writer.flush()

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

        # enforce move limit
        if move_count >= max_moves:
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
    # TensorFlow log for moves and outcome
    with tf_writer.as_default():
        tf.summary.scalar("SelfPlay/MovesPerGame", int(move_count), step=int(game_idx + 1))
        tf.summary.scalar("SelfPlay/Outcome", float(outcome), step=int(game_idx + 1))
    tf_writer.flush()
    logger.info("🧠 Logged moves and outcome for game %s to TensorBoard", game_idx + 1)
    for state, move in game_data:
        pass
    message = f"🏁 Game finished — {result_reason}. Moves: {len(game_data)} | Outcome: {outcome}"
    if outcome == 0.5:
        try:
            send_telegram_message("⚖️ Draw detected during self-play.")
        except Exception as e:
            logger.error("⚠️ Telegram send failed: %s", e)
    logger.debug("📨 Message to send: %s", message)
    try:
        send_telegram_message(message)
    except Exception as e:
        logger.error("⚠️ Telegram send failed: %s", e)

    if game_data:
        sample = game_data[0]
        try:
            send_telegram_message(f"🎯 Sample game generated.\nMoves: {len(game_data)} | First move index: {sample[1]}")
        except Exception as e:
            logger.error("⚠️ Telegram send failed: %s", e)
    else:
        logger.warning("⚠️ No game data generated to report.")
    logger.debug("🧠 RAM usage: %s%%", psutil.virtual_memory().percent)
    logger.info("✅ Game %s complete. Moves played: %s | Outcome: %s", game_idx + 1, len(game_data), outcome)
    if torch.cuda.is_available():
        logger.debug("💾 VRAM: %.2f MB", torch.cuda.memory_allocated(device) / 1024 ** 2)

    # Attach outcome to each move data
    game_data_with_outcome = [(state, move, outcome) for (state, move) in game_data]

    return game_idx, game_data_with_outcome


def self_play(num_games=100, sleep_time=0.0, max_moves=500):
    print("✅ self_play() function has started executing", flush=True)

    data = []

    # parallel self-play using multiprocessing Pool
    max_workers = min(num_games, os.cpu_count() or 1)
    model_path = os.path.join(BASE_DIR, "checkpoints", "model.pth")
    pool = mp.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(model_path, device.type, SEED)
    )
    tasks = [(idx, sleep_time, max_moves) for idx in range(num_games)]
    results = pool.starmap(_run_single_game, tasks)
    pool.close()
    pool.join()
    for idx, game_data in results:
        data.extend(game_data)

    # --- Self-play aggregated summary ---
    outcomes = [o for (_, _, o) in data]
    wins = sum(1 for o in outcomes if o > 0.5)
    draws = sum(1 for o in outcomes if o == 0.5)
    losses = sum(1 for o in outcomes if o < 0.5)
    # Log summary metrics to TensorBoard
    with tf_writer.as_default():
        tf.summary.scalar("SelfPlay/Wins", wins, step=0)
        tf.summary.scalar("SelfPlay/Draws", draws, step=0)
        tf.summary.scalar("SelfPlay/Losses", losses, step=0)
    tf_writer.flush()
    # Send Telegram summary notification
    try:
        send_telegram_message(f"📝 Self-play summary: 🏆 {wins} ⚖️ {draws} ❌ {losses}")
    except Exception as e:
        logger.error("⚠️ Telegram summary failed: %s", e)

    try:
        send_telegram_message(f"🤖 Starting self-play with {num_games} games...")
    except Exception as e:
        logger.error("⚠️ Telegram send failed: %s", e)
    logger.info("Starting self-play with %s games...", num_games)
    logger.debug("🧪 Self-play loop entered")
    try:
        send_telegram_message(f"🎮 Confirmed: Self-play function is running with {num_games} games.")
    except Exception as e:
        logger.error("⚠️ Telegram send failed: %s", e)

    logger.info("✅ Completed all self-play games: %s/%s", num_games, num_games)
    try:
        send_telegram_message(f"🏁 All {num_games} self-play games completed and logged to {SELFPLAY_LOG_DIR}.")
    except Exception:
        logger.error("⚠️ Telegram send failed for completion notice.")

    return data


# Piece value function for material evaluation
def piece_value(piece):
    values = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 0}
    return values.get(piece.upper(), 0)


def generate_self_play_data(model, num_games, device, sleep_time, max_moves=None):
    return self_play(model, num_games, device, sleep_time, max_moves=max_moves)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run self-play")
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"), help="Logging level")
    args = parser.parse_args()
    configure_logging(args.log_level)

    model = ChessNet()
    model_path = os.path.join(BASE_DIR, "checkpoints", "model.pth")
    if not os.path.exists(model_path):
        logger.error("❌ Model checkpoint not found: %s", model_path)
        exit(1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("✅ Model loaded successfully — starting self_play()")
    try:
        send_telegram_message("📦 Self-play started from __main__ with loaded model.")
    except Exception as e:
        logger.error("⚠️ Telegram send failed: %s", e)
    model.to(device)
    logger.info("✅ Loaded model from %s", model_path)
    data = self_play(num_games=50, max_moves=500)

    import json
    import numpy as np

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    save_path = os.path.join(BASE_DIR, f"self_play_data_{time.strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")
    with open(save_path, "w") as f:
        for state, move, outcome in data:
            f.write(json.dumps({
                "state": state,
                "move": move,
                "outcome": outcome
            }, cls=NumpyEncoder) + "\n")
    logger.info("💾 Saved self-play data to %s — %s samples.", save_path, len(data))

    logger.info("🧪 self_play() execution finished.")
    logger.info("Generated %s samples from self-play", len(data))

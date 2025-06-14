import sys
import os
import json
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging
import traceback
import zipfile
import psutil

from model import ChessNet
from self_play import self_play
from train import train_model

print("✅ Script loaded.", flush=True)

# === SETUP ===
try:
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = "/content/drive/MyDrive/KnightVision"
except ImportError:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

sys.excepthook = lambda exc_type, exc_value, exc_traceback: \
    print("Uncaught exception:", ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)), flush=True)

torch.set_default_tensor_type(torch.FloatTensor)

def load_or_initialize_model(model_path):
    model = ChessNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logger.info("✅ Loaded existing model.")
    else:
        logger.info("🆕 Initialized new model.")
    return model

def stream_human_data(file_path=os.path.join(DATA_DIR, "games.jsonl"), chunk_size=16, max_lines=1_000_000):
    with open(file_path, "r") as f:
        chunk = []
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def reinforcement_loop(iterations=3, games_per_iter=5, epochs=2):
    import time
    run_id = str(int(time.time()))
    log_dir = os.path.join(BASE_DIR, "runs", "chess_rl_v2", run_id)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    writer.add_scalar("Debug/Start", 1.0, 0)
    writer.flush()

    def cleanup_old_sessions(base_dir=os.path.join(BASE_DIR, "runs", "chess_rl_v2"), keep_last=3):
        subdirs = sorted(
            [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))],
            reverse=True
        )
        for dir_to_remove in subdirs[keep_last:]:
            import shutil
            logger.info(f"🧹 Cleaning up old session: {dir_to_remove}")
            shutil.rmtree(dir_to_remove)

    cleanup_old_sessions()

    global_step = 0
    model_path = os.path.join(checkpoint_dir, "model.pth")
    drive_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
    checkpoints_meta = []

    for i in range(iterations):
        logger.info(f"🚀 Iteration {i+1}/{iterations} - Generating self-play data")
        model = load_or_initialize_model(model_path)
        selfplay_data = self_play(model, num_games=games_per_iter)
        logger.info(f"🧠 Self-play generated {len(selfplay_data)} games")

        for human_chunk in stream_human_data(chunk_size=1):
            for sample in human_chunk:
                combined_data = selfplay_data + [sample]
                result = train_model(model, combined_data, epochs=epochs)
                avg_loss = sum(result['losses']) / len(result['losses'])

                writer.add_scalar("Training/Avg_Loss", avg_loss, global_step)
                writer.add_scalar("Training/SelfPlay_Size", len(selfplay_data), global_step)
                writer.add_scalar("Training/Combined_Sample_Size", len(combined_data), global_step)
                writer.flush()

                ckpt_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pth")
                torch.save(model.state_dict(), ckpt_path)
                torch.save(model.state_dict(), drive_checkpoint_path)
                checkpoints_meta.append((global_step, avg_loss))
                global_step += 1

        torch.save(model.state_dict(), model_path)
        logger.info(f"📦 Model saved after iteration {i+1}")

    # Save top checkpoints
    checkpoints_meta.sort(key=lambda x: x[1])  # sort by lowest loss
    best_steps = checkpoints_meta[:2]
    archive_path = os.path.join(checkpoint_dir, "archived_checkpoints.zip")
    with zipfile.ZipFile(archive_path, 'w') as z:
        for step, _ in best_steps:
            fname = f"model_step_{step}.pth"
            z.write(os.path.join(checkpoint_dir, fname), arcname=fname)

    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    torch.save(torch.load(os.path.join(checkpoint_dir, f"model_step_{best_steps[0][0]}.pth")), best_model_path)
    torch.save(torch.load(os.path.join(checkpoint_dir, f"model_step_{best_steps[0][0]}.pth")), drive_checkpoint_path)

    for step, _ in checkpoints_meta[2:]:
        path = os.path.join(checkpoint_dir, f"model_step_{step}.pth")
        if os.path.exists(path):
            os.remove(path)

    writer.close()
    logger.info("✅ Reinforcement learning complete.")

if __name__ == "__main__":
    logger.info("🎯 Starting full reinforcement training loop")
    reinforcement_loop(iterations=3, games_per_iter=5, epochs=2)
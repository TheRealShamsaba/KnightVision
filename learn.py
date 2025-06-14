
import sys
sys.path.append('/content/KnightVision/basicChess')  # for module resolution

import os
import json
import torch
from model import ChessNet
from self_play import self_play
from train import train_model
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging
import traceback
import zipfile
import psutil
from google.colab import drive

print("Script loaded", flush=True)

# === SETUP ===
drive.mount('/content/drive')
drive_checkpoint_dir = "/content/drive/MyDrive/KnightVision_Checkpoints"
os.makedirs(drive_checkpoint_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def handle_exception(exc_type, exc_value, exc_traceback):
    print("Uncaught exception:", ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)), flush=True)
sys.excepthook = handle_exception

torch.set_default_tensor_type(torch.FloatTensor)

def load_or_initialize_model(model_path):
    model = ChessNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logging.info("Loaded existing model.")
    else:
        logger.info("Initialized new model.")
    return model

def reinforcement_loop(iterations=3, games_per_iter=5, epochs=2):
    logger.info("Loading dataset from games.jsonl...")
    import time
    run_id = str(int(time.time()))
    log_dir = os.path.join("runs", "chess_rl_v2", run_id)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoints_meta = []

    def cleanup_old_sessions(base_dir="runs/chess_rl_v2", keep_last=3):
        subdirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))], reverse=True)
        for dir_to_remove in subdirs[keep_last:]:
            import shutil
            logger.info(f"Cleaning up old session: {dir_to_remove}")
            shutil.rmtree(dir_to_remove)

    cleanup_old_sessions()

    writer = SummaryWriter(log_dir)
    writer.add_scalar("Debug/Start", 1.0, 0)
    writer.flush()

    def stream_human_data(file_path="data/games.jsonl", chunk_size=16, max_lines=1_000_000):
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

    global_step = 0
    model_path = os.path.join(checkpoint_dir, "model.pth")

    for i in range(iterations):
        logger.info("=== Training Iteration %s/%s ===", i+1, iterations)
        model = load_or_initialize_model(model_path)
        data = self_play(model, num_games=games_per_iter)
        print(f"Self-play games: {len(data)}", flush=True)

        for human_chunk in stream_human_data(chunk_size=1):
            for sample in human_chunk:
                combined_sample = data + [sample]
                result = train_model(model, combined_sample, epochs=epochs)
                avg_loss = sum(result['losses']) / len(result['losses'])

                writer.add_scalar("Training/Avg_Loss", avg_loss, global_step)
                writer.add_scalar("Training/SelfPlay_Size", len(data), global_step)
                writer.add_scalar("Training/Sample_Size", len(combined_sample), global_step)
                writer.flush()

                ckpt_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pth")
                torch.save(model.state_dict(), ckpt_path)
                checkpoints_meta.append((global_step, avg_loss))

                torch.save(model.state_dict(), os.path.join(drive_checkpoint_dir, f"model_step_{global_step}.pth"))
                global_step += 1

        torch.save(model.state_dict(), model_path)
        logger.info("Model saved after iteration %s", i+1)

    # Archive and export top 2 checkpoints
    checkpoints_meta.sort(key=lambda x: x[1])
    top = checkpoints_meta[:2]
    archive_path = os.path.join(checkpoint_dir, "archived_checkpoints.zip")
    with zipfile.ZipFile(archive_path, 'w') as z:
        for step, _ in top:
            for suffix in ["pth", "txt"]:
                fname = f"model_step_{step}.{suffix}" if suffix == "pth" else f"checkpoint_info_{step}.txt"
                z.write(os.path.join(checkpoint_dir, fname), arcname=fname)

    best = top[0][0]
    best_path = os.path.join(checkpoint_dir, f"model_step_{best}.pth")
    torch.save(torch.load(best_path), os.path.join(checkpoint_dir, "best_model.pth"))
    torch.save(torch.load(best_path), os.path.join(drive_checkpoint_dir, "best_model.pth"))

    for step, _ in checkpoints_meta[2:]:
        os.remove(os.path.join(checkpoint_dir, f"model_step_{step}.pth"))
        txt = os.path.join(checkpoint_dir, f"checkpoint_info_{step}.txt")
        if os.path.exists(txt): os.remove(txt)

    writer.close()
    print("✅ Training complete.", flush=True)

if __name__ == "__main__":
    import time
    print("Starting training...", flush=True)
    time.sleep(3)
    reinforcement_loop(iterations=3, games_per_iter=5, epochs=2)
from model import ChessNet
from self_play import self_play
from train import train_model
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import glob

import logging
import sys
import traceback
import zipfile

print("Script loaded", flush=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

def handle_exception(exc_type, exc_value, exc_traceback):
    print("Uncaught exception:", ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)), flush=True)

sys.excepthook = handle_exception

# Use CPU by default for tensors
torch.set_default_tensor_type(torch.FloatTensor)  # Use CPU by default

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
    logger.info("Starting reinforcement loop with %d iterations...", iterations)

    import json
    def stream_human_data(file_path="data/games.jsonl", chunk_size=1, max_lines=100):
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

    import psutil
    global_step = 0
    model_path = os.path.join(checkpoint_dir, "model.pth")

    for i in range(iterations):
        logger.info("=== Training Iteration %s/%s ===", i+1, iterations)
        model = load_or_initialize_model(model_path)
        data = self_play(model, num_games=games_per_iter)
        print(f"Self-play games: {len(data)}", flush=True)

        for human_chunk in stream_human_data(chunk_size=1, max_lines=100):
            for sample in human_chunk:
                combined_sample = data + [sample]
                logger.info(f"Training on sample with self-play + 1 human move...")
                logger.info(f"Memory used: {psutil.Process().memory_info().rss / 1e6:.2f} MB")
                print(f"Training on {len(combined_sample)} samples (self-play + 1 human)", flush=True)
                print("Training started...", flush=True)
                result = train_model(model, combined_sample, epochs=epochs)
                logger.info("Train results: %s", result)

                writer.add_scalar("Training/Loss_Sample", sum(result['losses']) / len(result['losses']), global_step)
                writer.add_scalar("Training/Sample_Size", len(combined_sample), global_step)
                writer.add_scalar("Training/SelfPlay_Size", len(data), global_step)
                writer.add_scalar("Training/Avg_Loss", sum(result['losses']) / len(result['losses']), global_step)
                writer.add_scalar("Training/Min_Loss", min(result['losses']), global_step)
                writer.add_scalar("Training/Max_Loss", max(result['losses']), global_step)
                writer.flush()

                checkpoint_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                avg_loss = sum(result['losses']) / len(result['losses'])
                checkpoints_meta.append((global_step, avg_loss))
                with open(os.path.join(checkpoint_dir, f"checkpoint_info_{global_step}.txt"), "w") as meta_file:
                    meta_file.write(f"Step: {global_step}\n")
                    meta_file.write(f"Avg Loss: {avg_loss:.4f}\n")
                    meta_file.write(f"Min Loss: {min(result['losses']):.4f}\n")
                    meta_file.write(f"Max Loss: {max(result['losses']):.4f}\n")
                    meta_file.write(f"SelfPlay Samples: {len(data)}\n")
                    meta_file.write(f"Total Samples: {len(combined_sample)}\n")
                global_step += 1
        torch.save(model.state_dict(), model_path)
        logger.info("Model saved after iteration %s", i+1)

    checkpoints_meta.sort(key=lambda x: x[1])
    best_checkpoints = checkpoints_meta[:2]  # keep 2 best

    archive_path = os.path.join(checkpoint_dir, "archived_checkpoints.zip")
    with zipfile.ZipFile(archive_path, 'w') as zipf:
        for step, _ in best_checkpoints:
            model_file = os.path.join(checkpoint_dir, f"model_step_{step}.pth")
            meta_file = os.path.join(checkpoint_dir, f"checkpoint_info_{step}.txt")
            zipf.write(model_file, arcname=os.path.basename(model_file))
            zipf.write(meta_file, arcname=os.path.basename(meta_file))

    # Save best checkpoint model
    best_step, _ = best_checkpoints[0]
    best_model_path = os.path.join(checkpoint_dir, f"model_step_{best_step}.pth")
    torch.save(torch.load(best_model_path), os.path.join(checkpoint_dir, "best_model.pth"))

    # Clean up other checkpoints
    for step, _ in checkpoints_meta[2:]:
        os.remove(os.path.join(checkpoint_dir, f"model_step_{step}.pth"))
        os.remove(os.path.join(checkpoint_dir, f"checkpoint_info_{step}.txt"))

    writer.flush()
    print("✅ Logs flushed", flush=True)
    writer.close()


if __name__ == "__main__":
    print("Starting training process...", flush=True)
    print("Entering reinforcement_loop...", flush=True)
    import time
    print("Will sleep 5 seconds before training...", flush=True)
    time.sleep(5)
    reinforcement_loop(iterations=3, games_per_iter=5, epochs=2)
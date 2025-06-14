import sys
import os
def is_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False
import json
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging
import traceback
import zipfile
import psutil
import time
import gc
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Enabled memory growth for GPU: {gpu}")
    except Exception as e:
        print(f"⚠️ Could not enable memory growth for {gpu}: {e}")

# Set PyTorch default device and tensor type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Avoid global default override, let each tensor use `.to(device)`

from dotenv import load_dotenv
load_dotenv()

def format_duration(seconds):
    mins, secs = divmod(int(seconds), 60)
    hrs, mins = divmod(mins, 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

from model import ChessNet
from self_play import self_play
from train import train_model

print("✅ Script loaded.", flush=True)

# === SETUP ===
if is_colab():
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    BASE_DIR = "/content/drive/MyDrive/KnightVision"
else:
    base_dir_env = os.getenv("BASE_DIR")
    BASE_DIR = base_dir_env if base_dir_env else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

sys.excepthook = lambda exc_type, exc_value, exc_traceback: \
    print("Uncaught exception:", ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)), flush=True)


def load_or_initialize_model(model_path):
    model = ChessNet()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logger.info("✅ Loaded existing model.")
    else:
        logger.info("🆕 Initialized new model.")
    return model

def stream_human_data(file_path=os.path.join(DATA_DIR, "games.jsonl"), chunk_size=64, max_lines=1_000_000):
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
    total_start = time.time()
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

        human_batches_path = os.path.join(DATA_DIR, "human_batches")
        batch_files = sorted([
            os.path.join(human_batches_path, f) for f in os.listdir(human_batches_path)
            if f.startswith("games_part_") and f.endswith(".jsonl")
        ])

        logger.info(f"🧩 Total human batches: {len(batch_files)}")

        for batch_path in batch_files:
            import tensorflow as tf
            tf_logs = tf.summary.create_file_writer(log_dir)
            send_telegram_message(f"🧠 Starting training on batch file: {os.path.basename(batch_path)} (Step {global_step})")
            logger.info(f"📥 Loading human data from {batch_path}")
            with open(batch_path, "r") as f:
                human_data = [json.loads(line) for line in f]

            start_time = time.time()
            combined_data = selfplay_data + human_data
            result = train_model(
                model,
                combined_data,
                epochs=epochs,
                batch_size=2048,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                pin_memory=False
            )
            avg_loss = sum(result['losses']) / len(result['losses'])

            # --- Training score calculation and logging ---
            accuracy = result.get("accuracy", 0.0)
            reward = result.get("avg_reward", 0.0)
            score = (1 - avg_loss) * 50 + accuracy * 30 + reward * 20
            writer.add_scalar("Training/Score", score, global_step)
            print(f"📈 Score: {score:.2f}/100")

            writer.add_scalar("Training/Avg_Loss", avg_loss, global_step)
            # --- TensorFlow logging (for compatibility with TF tools) ---
            with tf_logs.as_default():
                tf.summary.scalar("Avg_Loss", avg_loss, step=global_step)
                tf.summary.scalar("Score", score, step=global_step)
                tf.summary.scalar("SelfPlay_Size", len(selfplay_data), step=global_step)
                tf.summary.scalar("Human_Batch_Size", len(human_data), step=global_step)
            writer.add_scalar("Training/SelfPlay_Size", len(selfplay_data), global_step)
            writer.add_scalar("Training/Human_Batch_Size", len(human_data), global_step)
            writer.flush()

            ckpt_path = os.path.join(checkpoint_dir, f"model_step_{global_step}.pth")
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), drive_checkpoint_path)
            checkpoints_meta.append((global_step, avg_loss))
            global_step += 1

            # --- Telegram notification block ---
            import random
            import requests

            telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
            telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

            def send_telegram_message(msg):
                if not telegram_token or not telegram_chat_id:
                    logger.warning("⚠️ Telegram token or chat ID not set.")
                    return
                url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
                data = {
                    "chat_id": telegram_chat_id,
                    "text": msg,
                    "parse_mode": "Markdown"
                }
                try:
                    requests.post(url, data=data)
                except Exception as e:
                    logger.warning(f"⚠️ Telegram send failed: {e}")

            fun_endings = [
                "💡 Fact: Magnus Carlsen once played 10 games at once — blindfolded.",
                "🕵️‍♂️ Tip: The engine is learning your favorite blunders.",
                "🎯 Goal: Defeat humanity by iteration 42.",
                "🧩 Every move counts. So does every update.",
                "👶 KnightVision IQ: now higher than a pigeon’s. Progress!"
            ]

            total_scalars = 0
            tf_event_files = [f for f in os.listdir(log_dir) if f.startswith("events.out.tfevents.")]
            if tf_event_files:
                for e in tf.compat.v1.train.summary_iterator(os.path.join(log_dir, tf_event_files[0])):
                    for v in e.summary.value:
                        total_scalars += 1

            batch_time = time.time() - start_time
            mem_used = psutil.Process(os.getpid()).memory_info().rss / 1e6  # in MB

            telegram_msg = (
                f"♟️ *KnightVision Training Report — Step {global_step}*\n"
                f"🏋️‍♂️ *Games Played:* {len(selfplay_data)} self-play | {len(human_data)} human\n"
                f"🧠 *Avg Loss:* `{avg_loss:.5f}` | 📉 Getting sharper!\n"
                f"🚀 *Step Time:* {format_duration(batch_time)}\n"
                f"💾 *RAM Used:* {mem_used:.2f} MB\n"
                f"📈 *TF Scalars Logged:* {total_scalars}\n"
                f"📦 *Model Saved:* Step_{global_step}.pth ✅\n\n"
                f"🧪 *Experiment:* Iteration {i+1}/{iterations}\n"
                f"🔥 Training with love, neurons, and caffeinated weights.\n"
                f"🧬 Stay tuned, the brain is evolving... 👾\n\n"
                + random.choice(fun_endings)
            )

            send_telegram_message(telegram_msg)
            send_telegram_message(f"✅ Completed training on {os.path.basename(batch_path)} at step {global_step}. Loss: {avg_loss:.5f}")
            # --- Progress alert every step ---
            if global_step % 1 == 0:
                send_telegram_message(f"📶 Progress ping: completed step {global_step}.")
            # --- End Telegram notification block ---

            logger.info(f"⏱️ Batch time: {format_duration(batch_time)} | RAM Used: {mem_used:.2f} MB")
            gc.collect()

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
    total_duration = time.time() - total_start
    logger.info(f"🕒 Total training time: {format_duration(total_duration)}")
    logger.info("✅ Reinforcement learning complete.")

if __name__ == "__main__":
    logger.info("🎯 Starting full reinforcement training loop")
    reinforcement_loop(iterations=3, games_per_iter=5, epochs=2)
import os
import sys
import multiprocessing as mp
import json
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import logging
import traceback
import sys
import zipfile
import psutil
import time
import gc
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    try:
        tf.config.set_memory_growth(gpu, True)
        print(f"✅ Enabled memory growth for GPU: {gpu}")
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        print(f"⚠️ Could not enable memory growth for {gpu}: {e}")
        sys.stdout.flush()
        sys.stderr.flush()


# === Google Colab Drive Mount ===
try:
    from google.colab import drive
    import sys
    if 'google.colab' in sys.modules:
        drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"⚠️ Skipping Colab mount: {e}")

# Set PyTorch default device and tensor type
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Running on device: {device}")
sys.stdout.flush()
sys.stderr.flush()
# Avoid global default override, let each tensor use `.to(device)`

from dotenv import load_dotenv
load_dotenv()

import random
import requests

from telegram_utils import send_telegram_message

# Helper to escape unsafe Markdown for Telegram
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

from model import ChessNet
from self_play import self_play
from train import train_model

print("✅ Script loaded.", flush=True)
sys.stdout.flush()
sys.stderr.flush()

BASE_DIR = "/content/drive/MyDrive/KnightVision"

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
        sys.stdout.flush()
        sys.stderr.flush()
    else:
        logger.info("🆕 Initialized new model.")
        sys.stdout.flush()
        sys.stderr.flush()
    print("[DEBUG] Model loaded and returned")
    sys.stdout.flush()
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

    model_path = os.path.join(checkpoint_dir, "model.pth")
    model = load_or_initialize_model(model_path)

    # Save initial model backup
    initial_model_path = os.path.join(CHECKPOINT_DIR, "initial_model.pth")
    torch.save(model.state_dict(), initial_model_path)
    print("💾 Initial model checkpoint saved.")
    sys.stdout.flush()
    sys.stderr.flush()
    # Ensure send_telegram_message is imported or defined above
    # (Already imported at top of file)
    send_telegram_message("💾 Initial model checkpoint saved.")

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
            sys.stdout.flush()
            sys.stderr.flush()
            shutil.rmtree(dir_to_remove)

    cleanup_old_sessions()

    global_step = 0
    drive_checkpoint_path = os.path.join(CHECKPOINT_DIR, "model_latest.pth")
    checkpoints_meta = []

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"🚨 DEBUG: token={os.getenv('TELEGRAM_BOT_TOKEN')}, chat_id={os.getenv('TELEGRAM_CHAT_ID')}")
    sys.stdout.flush()
    sys.stderr.flush()
    print("🤖 Starting KnightVision RL — token and chat ID loaded successfully.")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        send_telegram_message("🤖 Starting KnightVision RL — token and chat ID loaded successfully.")
    except Exception as e:
        print(f"⚠️ Telegram failed to notify start: {e}")

    for i in range(iterations):
        msg_iter = f"🌀 Iteration {i+1}/{iterations} started..."
        try:
            send_telegram_message(msg_iter)
        except Exception as e:
            print(f"⚠️ Telegram failed: {e}")
        print("📨 Sent message:", msg_iter)
        print(f"🌀 Iteration {i+1}/{iterations} started...", flush=True)
        print(f"[INFO] Starting self-play iteration {i+1} of {iterations}")
        sys.stdout.flush()
        logger.info(f"🚀 Iteration {i+1}/{iterations} - Generating self-play data")
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            print("🔁 Self-play starting...", flush=True)
            print(f"💾 Current training epoch: {i+1}", flush=True)
            sys.stdout.flush()
            sys.stderr.flush()
            selfplay_data = self_play(model, num_games=games_per_iter)
            print(f"✅ Self-play completed. Games collected: {len(selfplay_data)}", flush=True)
            print(f"[DEBUG] self_play() returned {len(selfplay_data)} samples")
            sys.stdout.flush()
            if selfplay_data:
                print("[DEBUG] First sample from self_play:", selfplay_data[0])
                sys.stdout.flush()
            # === DEBUG BLOCK: print number of samples ===
            print(f"✅ Self-play returned {len(selfplay_data)} samples")
            sys.stdout.flush()
            sys.stderr.flush()
            if len(selfplay_data) == 0:
                msg_zero = "⚠️ Self-play returned 0 samples — training skipped."
                try:
                    send_telegram_message(msg_zero)
                except Exception as e:
                    print(f"⚠️ Telegram failed: {e}")
                print("📨 Sent message:", msg_zero)
            print(f"🧪 Generated {len(selfplay_data)} self-play games")
            sys.stdout.flush()
            sys.stderr.flush()
            if selfplay_data:
                print("🔍 First self-play sample:", selfplay_data[0])
                sys.stdout.flush()
                sys.stderr.flush()
            logger.info(f"🧠 Self-play generated {len(selfplay_data)} games")
            sys.stdout.flush()
            sys.stderr.flush()
            if len(selfplay_data) == 0:
                logger.warning("⚠️ Self-play returned 0 games. This may indicate a bug.")
                sys.stdout.flush()
                sys.stderr.flush()
                msg_zero2 = "⚠️ Self-play returned 0 games. Please inspect the logic."
                try:
                    send_telegram_message(msg_zero2)
                except Exception as e:
                    print(f"⚠️ Telegram failed: {e}")
                print("📨 Sent message:", msg_zero2)
            else:
                logger.info("✅ Self-play completed successfully.")
                sys.stdout.flush()
                sys.stderr.flush()
                msg_selfplay_complete = f"♟️ Self-play complete — {len(selfplay_data)} games generated."
                try:
                    send_telegram_message(msg_selfplay_complete)
                except Exception as e:
                    print(f"⚠️ Telegram failed: {e}")
                print("📨 Sent message:", msg_selfplay_complete)
                print("♟️ Self-play finished with", len(selfplay_data), "games.")
                # Optionally print first game
                logger.debug(f"🔍 Sample self-play game: {selfplay_data[0]}")
                sys.stdout.flush()
                sys.stderr.flush()
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"🔥 Self-play crashed: {e}\n{error_details}")
            sys.stdout.flush()
            sys.stderr.flush()
            msg_crash = f"🔥 Self-play crashed with error:\n{e}"
            try:
                send_telegram_message(msg_crash)
            except Exception as te:
                print(f"⚠️ Telegram failed: {te}")
            print("📨 Sent message:", msg_crash)
            print("❌ Exception occurred during self-play:", e)
            raise e

        human_batches_path = os.path.join(DATA_DIR, "human_batches")
        import errno

        try:
            batch_files = sorted([
                os.path.join(human_batches_path, f)
                for f in os.listdir(human_batches_path)
                if f.startswith("games_part_") and f.endswith(".jsonl")
            ])
        except OSError as e:
            if e.errno == errno.EIO:
                print(f"⚠️ Google Drive I/O error: {e}. Skipping human batch loading this round.")
                batch_files = []
            else:
                raise

        logger.info(f"🧩 Total human batches: {len(batch_files)}")
        sys.stdout.flush()
        sys.stderr.flush()

        for batch_path in batch_files:
            import tensorflow as tf
            tf_logs = tf.summary.create_file_writer(log_dir)
            send_telegram_message(f"🧠 Starting training on batch file: {os.path.basename(batch_path)} (Step {global_step})")
            send_telegram_message("🚀 Beginning data loading and collation...")
            logger.info(f"📥 Loading human data from {batch_path}")
            sys.stdout.flush()
            sys.stderr.flush()
            with open(batch_path, "r") as f:
                human_data = [json.loads(line) for line in f]

            start_time = time.time()
            combined_data = selfplay_data + human_data
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                logger.info(f"🧠 GPU Mem before training: {info.used / 1e6:.2f} MB used")
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception as e:
                logger.warning(f"⚠️ GPU monitoring failed: {e}")
                sys.stdout.flush()
                sys.stderr.flush()

            result = train_model(
                model,
                combined_data,
                epochs=epochs,
                batch_size=2048,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                pin_memory=False
            )
            avg_loss = sum(result['losses']) / len(result['losses'])
            print(f"📤 Training step {global_step} complete. Avg Loss: {avg_loss:.5f}")
            sys.stdout.flush()
            sys.stderr.flush()
            msg_train_complete = f"📤 Training step {global_step} complete. Avg Loss: {avg_loss:.5f}"
            try:
                send_telegram_message(msg_train_complete)
            except Exception as e:
                print(f"⚠️ Telegram failed: {e}")
            print("📨 Sent message:", msg_train_complete)

            # --- Training score calculation and logging ---
            accuracy = result.get("accuracy", 0.0)
            reward = result.get("avg_reward", 0.0)
            score = (1 - avg_loss) * 50 + accuracy * 30 + reward * 20
            writer.add_scalar("Training/Score", score, global_step)
            print(f"📈 Score: {score:.2f}/100")
            sys.stdout.flush()
            sys.stderr.flush()

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
            # Notify after checkpoint save
            try:
                send_telegram_message(f"💾 Model checkpoint saved to: {ckpt_path}")
            except Exception as e:
                print(f"⚠️ Telegram failed to notify checkpoint save: {e}")
            msg_ckpt = f"💾 Model checkpoint saved at step {global_step}"
            try:
                send_telegram_message(msg_ckpt)
            except Exception as e:
                print(f"⚠️ Telegram failed: {e}")
            print("📨 Sent message:", msg_ckpt)
            # Periodic autosave to backup file
            if global_step % 2 == 0:
                autosave_path = os.path.join(CHECKPOINT_DIR, "autosave_model.pth")
                torch.save(model.state_dict(), autosave_path)
                # Notify after autosave
                try:
                    send_telegram_message("💾 Autosave checkpoint saved to main directory.")
                except Exception as e:
                    print(f"⚠️ Telegram failed to notify autosave: {e}")
                msg_autosave = "💾 Autosave model checkpoint saved."
                try:
                    send_telegram_message(msg_autosave)
                except Exception as e:
                    print(f"⚠️ Telegram failed: {e}")
                print("📨 Sent message:", msg_autosave)
            with open(os.path.join(checkpoint_dir, f"model_step_{global_step}.txt"), 'w') as ts_file:
                ts_file.write(f"Checkpoint saved at step {global_step}")
            torch.save(model.state_dict(), drive_checkpoint_path)
            checkpoints_meta.append((global_step, avg_loss))
            global_step += 1

            # --- Telegram notification block ---

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
                f"📈 *KnightVision TF Log Update — Step {global_step}*\n"
                f"🧠 *Avg Loss:* `{avg_loss:.5f}`\n"
                f"📊 *TF Scalars Logged:* {total_scalars}\n"
                f"🕒 *Step Time:* {format_duration(batch_time)}\n"
                f"💾 *RAM Used:* {mem_used:.2f} MB\n"
            )
            print("📨 Telegram message preview:\n", telegram_msg)
            sys.stdout.flush()
            sys.stderr.flush()

            try:
                send_telegram_message(telegram_msg)
            except Exception as e:
                print(f"⚠️ Telegram failed: {e}")
            print("📨 Sent message:", telegram_msg)
            msg_completed = f"✅ Completed training on {os.path.basename(batch_path)} at step {global_step}. Loss: {avg_loss:.5f}"
            try:
                send_telegram_message(msg_completed)
            except Exception as e:
                print(f"⚠️ Telegram failed: {e}")
            print("📨 Sent message:", msg_completed)
            msg_uploaded = f"📤 Uploaded model checkpoint for step {global_step}. Ready for next batch."
            try:
                send_telegram_message(msg_uploaded)
            except Exception as e:
                print(f"⚠️ Telegram failed: {e}")
            print("📨 Sent message:", msg_uploaded)
            # --- Progress alert every step ---
            notify_every = 1
            if global_step % notify_every == 0:
                msg_progress = f"📶 Progress ping: completed step {global_step}."
                try:
                    send_telegram_message(msg_progress)
                except Exception as e:
                    print(f"⚠️ Telegram failed: {e}")
                print("📨 Sent message:", msg_progress)
            # --- End Telegram notification block ---

            logger.info(f"⏱️ Batch time: {format_duration(batch_time)} | RAM Used: {mem_used:.2f} MB")
            sys.stdout.flush()
            sys.stderr.flush()
            gc.collect()

        torch.save(model.state_dict(), model_path)
        with open(os.path.join(checkpoint_dir, f"model_step_{global_step}.txt"), 'w') as ts_file:
            ts_file.write(f"Checkpoint saved at step {global_step}")
        logger.info(f"📦 Model saved after iteration {i+1}")
        sys.stdout.flush()
        sys.stderr.flush()
        # Notify via Telegram after training step and model save
        from telegram_utils import send_telegram_message
        send_telegram_message("🤖 Training completed successfully and model updated.")

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
    print("🏁 Reinforcement training loop has finished.")
    sys.stdout.flush()
    sys.stderr.flush()
    msg_rl_finished = "🏁 Reinforcement training loop has finished."
    try:
        send_telegram_message(msg_rl_finished)
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")
    print("📨 Sent message:", msg_rl_finished)
    msg_final_ckpt = "🧠 Final model checkpoint saved to Drive."
    try:
        send_telegram_message(msg_final_ckpt)
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")
    print("📨 Sent message:", msg_final_ckpt)
    total_duration = time.time() - total_start
    with open(os.path.join(BASE_DIR, "last_training_summary.txt"), 'w') as f:
        f.write(f"Training completed in {format_duration(total_duration)}\nBest steps: {best_steps}")
        msg_summary = f"📊 Training complete. Total time: {format_duration(total_duration)}. Best checkpoint: step {best_steps[0][0]}"
        try:
            send_telegram_message(msg_summary)
        except Exception as e:
            print(f"⚠️ Telegram failed: {e}")
        print("📨 Sent message:", msg_summary)
    # Backup final model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "final_model.pth"))
    logger.info(f"🕒 Total training time: {format_duration(total_duration)}")
    logger.info("✅ Reinforcement learning complete.")
    sys.stdout.flush()
    sys.stderr.flush()

# --- Main entry point for training ---
def main():
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    reinforcement_loop()

if __name__ == "__main__":
    main()
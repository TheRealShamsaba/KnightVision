print("Training script loaded...")
import sys
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F
import tensorflow as tf
from self_play import generate_self_play_data
import argparse
# --- Evaluation function ---
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for boards_np, moves, outcomes in data_loader:
            boards = boards_np.float().to(device)
            moves = moves.to(device)
            outcomes = outcomes.to(device)
            preds_policy, preds_value = model(boards)
            loss_policy = F.cross_entropy(preds_policy.float(), moves)
            loss_value = F.mse_loss(preds_value.squeeze().float(), outcomes)
            total_loss += (loss_policy + loss_value).item() * boards.size(0)
            total_samples += boards.size(0)
    model.train()
    return total_loss / total_samples if total_samples > 0 else float('inf')

def _train_one_epoch(model, dataloader, optimizer, epoch, device, writer, scaler, REWARD_SHAPING_COEF, ENTROPY_COEF, accumulate_steps: int = 1):
    model.train()
    total_loss = 0
    total_reward = 0
    dataloader_iter = iter(dataloader)
    loss_policy = torch.tensor(0.0)
    loss_value = torch.tensor(0.0)
    preds_policy = torch.tensor([])
    preds_value = torch.tensor([])
    last_moves = None
    optimizer.zero_grad()
    num_batches = len(dataloader)
    for i in range(num_batches):
        try:
            boards_np, moves, outcomes = next(dataloader_iter)
            print(f"🔁 Processing batch {i+1}/{num_batches}")
            last_moves = moves
        except Exception as e:
            print(f"⚠️ Data loading error: {e}")
            continue

        boards = boards_np.float()
        moves = moves.long()
        outcomes = outcomes.float()

        boards = boards.to(device)
        moves = moves.to(device)
        outcomes = outcomes.to(device)
        # Use discrete self-play outcomes directly as reward (+1, 0, -1)
        total_reward += outcomes.sum().item()
        writer.add_scalar("Metrics/Reward", outcomes.sum().item(), epoch * num_batches + i)

        with torch.cuda.amp.autocast():
            preds_policy, preds_value = model(boards)

        if preds_policy.size(0) == moves.size(0):
            _, predicted_moves = torch.max(preds_policy, 1)
            batch_accuracy = (predicted_moves == moves).float().mean().item()
        else:
            batch_accuracy = 0.0

        loss_policy = F.cross_entropy(preds_policy.float(), moves)
        # Value head is trained to predict the final reward (outcome)
        loss_value = F.mse_loss(preds_value.squeeze().float(), outcomes)
        # Entropy regularization
        log_probs = F.log_softmax(preds_policy.float(), dim=1)
        probs = F.softmax(preds_policy.float(), dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss = loss_policy + loss_value - ENTROPY_COEF * entropy

        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️ Skipping batch due to invalid loss (NaN or Inf)")
            continue

        # Gradient accumulation: scale loss by accumulate_steps
        loss = loss / accumulate_steps
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Only step optimizer every accumulate_steps batches, or at the end
        if ((i + 1) % accumulate_steps == 0) or (i == num_batches - 1):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulate_steps  # restore loss to original scale for logging

        if i % 5 == 0:
            print(f"Epoch {epoch+1} | Batch {i+1}/{num_batches} | Loss: {loss.item() * accumulate_steps:.4f} | GPU Mem: {torch.cuda.memory_allocated(device) / 1e6:.1f}MB")
    return total_loss, loss_policy, loss_value, preds_policy, preds_value, last_moves, total_reward

def _run_validation(model, val_loader, device, writer, epoch, plateau_scheduler, checkpoint_dir, best_val_loss, epochs_no_improve, PATIENCE, send_telegram_message):
    val_loss = evaluate(model, val_loader, device)
    writer.add_scalar("Val/Loss", val_loss, epoch)
    plateau_scheduler.step(val_loss)
    send_telegram_message(f"📊 Validation Loss: {val_loss:.4f}")
    early_stop = False
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        from datetime import datetime
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f"/content/drive/MyDrive/KnightVision/checkpoints/best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
        send_telegram_message(f"✅ New best model saved with val loss {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            send_telegram_message(f"⏹️ Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
            early_stop = True
    return val_loss, best_val_loss, epochs_no_improve, early_stop

def _run_self_play(model, num_selfplay_games, device, sleep_time, max_moves, dataset, batch_size, pin_memory, num_workers, DataLoader):
    print("♟️ Generating self-play games...")
    # Call generate_self_play_data using positional arguments only
    new_selfplay_data = generate_self_play_data(
        model, num_selfplay_games, device, max_moves
    )
    print(f"✅ Self-play data generated: {len(new_selfplay_data)} samples")
    import time; time.sleep(sleep_time)
    dataloader = None
    if new_selfplay_data:
        if hasattr(dataset, "extend") and callable(dataset.extend):
            dataset.extend(new_selfplay_data)
            loader_kwargs = {
                "batch_size": batch_size,
                "pin_memory": pin_memory,
                "num_workers": num_workers,
            }
            dataloader = DataLoader(
                dataset,
                shuffle=True,
                **(
                    {"persistent_workers": True, "prefetch_factor": 2}
                    if num_workers > 0 else {}
                ),
                **loader_kwargs,
            )
            print(f"✅ {len(new_selfplay_data)} self-play games successfully added to dataset and new DataLoader created.")
        else:
            print("⚠️ Dataset does not support extension method; self-play data ignored.")
    else:
        print("⚠️ No self-play games generated.")
    return dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=5, help="Number of total epochs to train")
args = parser.parse_args()

num_epochs = args.epochs
# === New train_with_validation function ===
def train_with_validation(model, optimizer, start_epoch, train_dataset, val_dataset, epochs=num_epochs, batch_size=2048, device='cpu', pin_memory=False, num_workers=0):
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()
    # Build DataLoader kwargs
    loader_kwargs = {"batch_size": batch_size}
    if pin_memory:
        loader_kwargs["pin_memory"] = True
    if num_workers > 0:
        loader_kwargs["num_workers"] = num_workers
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    print(f"✅ DataLoaders created: {len(train_dataset)} train, {len(val_dataset)} val samples")
    dataloader = train_loader
    dataset = train_dataset
    # Number of epochs to use PGN data only before introducing self-play
    NUM_PGN_EPOCHS = 5  # Number of epochs to use PGN data only
    writer = SummaryWriter(log_dir=os.path.join(BASE_DIR, "runs", run_name))
    tf_log_dir = os.path.join(BASE_DIR, "runs", run_name, "tf_logs")
    tf_writer = tf.summary.create_file_writer(tf_log_dir)
    print(f"Logging to: runs/{run_name}")
    model.train()
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=int(os.getenv("COSINE_T0", "10")),
        T_mult=1
    )
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_GAMMA,
        patience=PATIENCE,
        verbose=True
    )
    model.to(device)
    torch.backends.cudnn.benchmark = True
    all_losses = []
    all_rewards = []
    all_accuracies = []
    all_scores = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    # Quick test: disable self-play entirely
    num_selfplay_games = int(os.getenv("NUM_SELFPLAY_GAMES", "0"))    
    sleep_time = 0.0
    max_moves = None
    print("Starting training...")
    send_telegram_message("✅ train.py started training...")
    print("✅ Starting epoch loop...")
    last_moves = None
    # Get accumulate_steps from environment
    accumulate_steps = int(os.getenv("ACCUM_STEPS", "1"))
    for epoch in range(start_epoch, epochs):
        send_telegram_message(f"🚀 Starting epoch {epoch+1}")
        if (epoch + 1) % 10 == 0:
            from datetime import datetime
            torch.save(
                model.state_dict(),
                f"/content/drive/MyDrive/KnightVision/checkpoints/model_epoch_{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': None,
            }, f"/content/drive/MyDrive/KnightVision/checkpoints/checkpoint_epoch_LAST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            send_telegram_message(f"📦 Checkpoint saved — Epoch {epoch+1}")

        # --- DataLoader selection: PGN only for first NUM_PGN_EPOCHS epochs ---
        if epoch < NUM_PGN_EPOCHS:
            dataloader = train_loader  # Use PGN data only
            print(f"✅ Epoch {epoch+1}: Using PGN dataset only (pre-training).")
        else:
            # After NUM_PGN_EPOCHS, use self-play data if available
            new_dataloader = None
            new_dataloader = _run_self_play(
                model=model,
                num_selfplay_games=num_selfplay_games,
                device=device,
                sleep_time=sleep_time,
                max_moves=max_moves,
                dataset=dataset,
                batch_size=batch_size,
                pin_memory=pin_memory,
                num_workers=num_workers,
                DataLoader=DataLoader
            )
            if new_dataloader is not None:
                dataloader = new_dataloader  # Switch to self-play if available
            else:
                dataloader = train_loader
            print(f"✅ Epoch {epoch+1}: Using self-play data mixed in.")

        # --- Training ---
        total_loss, loss_policy, loss_value, preds_policy, preds_value, last_moves, total_reward = _train_one_epoch(
            model, dataloader, optimizer, epoch, device, writer, scaler, REWARD_SHAPING_COEF, ENTROPY_COEF, accumulate_steps=accumulate_steps
        )
        # --- Validation & Early-Stopping ---
        val_loss, best_val_loss, epochs_no_improve, early_stop = _run_validation(
            model, val_loader, device, writer, epoch, plateau_scheduler, checkpoint_dir, best_val_loss, epochs_no_improve, PATIENCE, send_telegram_message
        )
        if early_stop:
            break
        # --- Logging ---
        writer.add_scalar("Loss/Total", total_loss, epoch)
        writer.add_scalar("Loss/Policy", loss_policy.item(), epoch)
        writer.add_scalar("Loss/Value", loss_value.item(), epoch)
        if last_moves is not None and preds_policy.size(0) == last_moves.size(0) and preds_policy.size(0) != 0:
            _, predicted_moves = torch.max(preds_policy, 1)
            last_moves = last_moves.to(predicted_moves.device)
            accuracy = (predicted_moves == last_moves).float().mean().item()
        else:
            accuracy = 0.0
        writer.add_scalar("Metrics/Accuracy", accuracy, epoch)
        # avg_reward: average reward per sample in this epoch (total_reward is sum of game outcomes)
        avg_reward = total_reward / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
        writer.add_scalar("Metrics/AvgReward", avg_reward, epoch)
        # score uses total_reward (sum of game outcomes) scaled by dataset size
        score = (accuracy * 100) - (total_loss * 0.5) + (avg_reward * 10)
        score = max(0, min(score, 100))
        writer.add_scalar("Metrics/TrainingScore", score, epoch)
        with tf_writer.as_default():
            tf.summary.scalar("Loss/Total", total_loss, step=epoch)
            tf.summary.scalar("Metrics/Accuracy", accuracy, step=epoch)
            tf.summary.scalar("Hyperparams/LearningRate", optimizer.param_groups[0]["lr"], step=epoch)
        print(f"\n🧠 Training Report — Epoch {epoch+1}")
        print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
        print(f"📉 Loss: {total_loss:.4f}")
        print(f"🏋️‍♂️ Avg Reward: {avg_reward:.4f} (total_reward = sum of game outcomes in this epoch)")
        print(f"📈 Score: {score:.2f}/100\n")
        summary_msg = (
            f"🏁 Epoch {epoch+1} finished\n"
            f"🎯 Accuracy: {accuracy * 100:.2f}%\n"
            f"📉 Loss: {total_loss:.4f}\n"
            f"🏋️‍♂️ Avg Reward: {avg_reward:.4f} (total_reward = sum of game outcomes in this epoch)\n"
            f"📈 Score: {score:.2f}/100"
        )
        send_telegram_message(summary_msg)
        writer.flush()
        cos_scheduler.step(epoch + 1)
        writer.add_scalar("Hyperparams/LearningRate", optimizer.param_groups[0]["lr"], epoch)
        all_losses.append(total_loss)
        all_rewards.append(avg_reward)
        all_accuracies.append(accuracy)
        all_scores.append(score)
        if preds_policy.numel() > 0:
            writer.add_histogram("Distributions/Policy", preds_policy, epoch)
        if preds_value.numel() > 0:
            writer.add_histogram("Distributions/Value", preds_value, epoch)
        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, epoch)
        logger.info(
            "Epoch %s/%s - Loss: %.4f - avg Reward: %.4f (total_reward = sum of game outcomes)",
            epoch + 1,
            epochs,
            total_loss,
            avg_reward,
        )
    model.eval()
    writer.flush()
    writer.close()
    message = (
        "🏁 *train.py finished training.*\n"
        "All epochs completed successfully. Check TensorBoard for metrics and the checkpoints folder for saved models."
    )
    send_telegram_message(message)
    return {
        "losses": all_losses,
        "rewards": all_rewards,
        "accuracies": all_accuracies,
        "scores": all_scores
    }
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Ensure NUM_SELFPLAY_GAMES is set in the environment with a default of "50"
os.environ["NUM_SELFPLAY_GAMES"] = os.getenv("NUM_SELFPLAY_GAMES", "50")


# === RL hyperparameters ===
ENTROPY_COEF = float(os.getenv("ENTROPY_COEF", "0.01"))
REWARD_SHAPING_COEF = float(os.getenv("REWARD_SHAPING_COEF", "1.0"))
LR_STEP_SIZE = int(os.getenv("LR_STEP_SIZE", "10"))
LR_GAMMA = float(os.getenv("LR_GAMMA", "0.1"))
IN_COLAB = False
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
run_name = "chess_rl_v2"
checkpoint_dir = os.path.join(BASE_DIR, "runs", run_name, "checkpoints")
resume_checkpoint = os.path.join(checkpoint_dir, "checkpoint_epoch_LAST.pth")
os.makedirs(checkpoint_dir, exist_ok=True)
import torch
try:
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("✅ TensorFlow GPU memory growth enabled")
except Exception as e:
    print(f"⚠️ TensorFlow GPU config failed: {e}")
import torch.nn
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
import json
import numpy as np

# === Deterministic seeding for reproducibility ===
SEED = int(os.getenv("SEED", "42"))
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    tf.random.set_seed(SEED)
except NameError:
    pass
import chess
import chess.pgn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from datetime import datetime
from telegram_utils import send_telegram_message
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Do not set multiprocessing start method globally here; move to main block.
from torch.utils.tensorboard import SummaryWriter

# Validation & Early-Stopping config
PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "5"))

logger = logging.getLogger(__name__)

class ChessPGNDataset(Dataset):
    def __init__(self, path=os.path.join(BASE_DIR, "data", "games.jsonl"), move_encoder=None, max_samples=10000):
        self.file_path = path
        self.move_encoder = move_encoder or self.default_move_encoder
        self.max_samples = max_samples
        self.additional_data = []

        # Store only the byte offsets for each line for sampling
        self.line_offsets = []
        with open(self.file_path, 'rb') as f:
            offset = 0
            for i, line in enumerate(f):
                if i >= self.max_samples:
                    break
                self.line_offsets.append(offset)
                offset += len(line)
        print(f"✅ ChessPGNDataset loaded: {len(self.line_offsets)} samples found in {self.file_path}")

    def __len__(self):
        return len(self.line_offsets) + len(self.additional_data)
    
    def __getitem__(self, idx):
        if idx >= len(self.line_offsets):
            return self.additional_data[idx - len(self.line_offsets)]
        offset = self.line_offsets[idx]
        with open(self.file_path, 'r') as f:
            f.seek(offset)
            record = json.loads(f.readline().strip())

        fen = record["fen"]
        move_san = record["move"]
        board_tensor = self.fen_to_tensor(fen)
        move_index = self.move_encoder(move_san, fen)

        outcome = 1.0 if chess.Board(fen).turn == chess.WHITE else -1.0
        return board_tensor, move_index, outcome

    def fen_to_tensor(self, fen):
        
        import numpy as np
        board = chess.Board(fen)
        piece_map = board.piece_map()
        tensor = np.zeros((12,8,8), dtype=np.float32)
        piece_to_plane = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for square, piece in piece_map.items():
            plane = piece_to_plane[piece.symbol()]
            row = 7 - (square // 8)
            col = square % 8 
            tensor[plane][row][col] = 1.0
        return tensor
    def default_move_encoder(self, move_san, fen):
        board = chess.Board(fen)
        move = board.parse_san(move_san)
        from_square = move.from_square
        to_square = move.to_square
        return from_square * 64 + to_square

    def extend(self, new_records):
        """
        Extend dataset with additional self-play records.
        new_records: list of (board_tensor, move_index, outcome) tuples.
        """
        self.additional_data.extend(new_records)
from model_utils import load_or_initialize_model
from model import ChessNet

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Custom collate function for DataLoader
def custom_collate(batch):
    boards, moves, outcomes = zip(*batch)
    boards = torch.from_numpy(np.stack(boards)).float().contiguous()
    moves = torch.tensor(moves).long()
    outcomes = torch.tensor(outcomes).float()
    if device.type == "cuda":
        return boards.to(device, non_blocking=True), moves.to(device, non_blocking=True), outcomes.to(device, non_blocking=True)
    else:
        return boards, moves, outcomes

import sys
from model_utils import load_or_initialize_model

games_path = "/content/drive/MyDrive/KnightVision/data/games.jsonl"
# Check if file exists and is non-empty before proceeding
if not os.path.isfile(games_path) or os.path.getsize(games_path) == 0:
    msg = f"❌ Dataset file not found or empty: {games_path}"
    print(msg)
    send_telegram_message(msg)
    print("✅ Telegram message sent.")
    sys.exit(1)
 # Initialize model, optimizer, and start_epoch using unified loader
model, optimizer, start_epoch = load_or_initialize_model(
    model_class=ChessNet,
    optimizer_class=optim.Adam,
    optimizer_kwargs={'lr': 1e-3},
    model_path=resume_checkpoint,
    device=device
)
# Enable multi-GPU data parallelism if available
if torch.cuda.device_count() > 1:
    logger.info(f"🌐 Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
print("✅ Model initialized")
training_dataset = ChessPGNDataset(games_path, max_samples=1000000)
print(f"✅ Dataset instantiated: {len(training_dataset)} samples")
if len(training_dataset) == 0:
    msg = f"❌ Dataset loaded but contains 0 samples: {games_path}"
    print(msg)
    send_telegram_message(msg)
    print("✅ Telegram message sent.")
    sys.exit(1)

print("✅ Dataset ready")


# Split dataset into training and validation sets
val_ratio = float(os.getenv("VAL_RATIO", "0.1"))
val_size = int(len(training_dataset) * val_ratio)
train_size = len(training_dataset) - val_size
train_dataset, validation_dataset = random_split(training_dataset, [train_size, val_size])
print(f"✅ Dataset split: {train_size} train, {val_size} val samples")





# Send notification that training started

# Notify that training has started

send_telegram_message("🚀 Training started...")


# Function to capture stdout and stderr during training and send via Telegram in chunks
def capture_and_train():
    # assuming validation_dataset is prepared earlier (e.g., split from training_dataset)
    print(f"🔧 Training: epochs={args.epochs}, batch_size=512, pin_memory=False, num_workers=4")
    try:
        result = train_with_validation(
            model=model,
            optimizer=optimizer,
            start_epoch=start_epoch,
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            epochs=args.epochs,
            batch_size=512,
            device=device,
            pin_memory=False,
            num_workers=4
        )
        print("✅ Training complete: model saved to best_model.pth")
    except Exception as e:
        msg = f"❌ Training failed: {e}"
        print(msg)
        send_telegram_message(msg)
        print("✅ Telegram message sent.")
        return
    # Send a short summary first
    summary = ""
    if isinstance(result, dict) and result.get("losses") and result.get("accuracies"):
        summary = (
            f"✅ Training completed.\n"
            f"Final Loss: {result['losses'][-1]:.4f}\n"
            f"Final Accuracy: {result['accuracies'][-1]*100:.2f}%\n"
            f"Final Score: {result['scores'][-1]:.2f}\n"
            f"Check TensorBoard and checkpoints for details."
        )
    else:
        summary = "✅ Training completed. Check logs for details."
    send_telegram_message(summary)
    print("✅ Telegram message sent.")

# Main entry point for script execution
if __name__ == '__main__':
    import torch.multiprocessing
    if not IN_COLAB:
        torch.multiprocessing.set_start_method('spawn', force=True)
    # Define helper functions only in main process to avoid multiprocessing issues
    globals().update({
        "_train_one_epoch": _train_one_epoch,
        "_run_validation": _run_validation,
        "_run_self_play": _run_self_play,
    })
    capture_and_train()
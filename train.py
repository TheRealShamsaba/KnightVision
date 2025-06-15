print("Training script loaded...")
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
# Ensure NUM_SELFPLAY_GAMES is set in the environment with a default of "50"
os.environ["NUM_SELFPLAY_GAMES"] = os.getenv("NUM_SELFPLAY_GAMES", "50")
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
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
import json
import numpy as np
import chess
import chess.pgn
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from self_play import generate_self_play_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
## Do not set multiprocessing start method globally here; move to main block.
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class ChessPGNDataset(Dataset):
    def __init__(self, path=os.path.join(BASE_DIR, "data", "games.jsonl"), move_encoder=None, max_samples=10000):
        self.file_path = path
        self.move_encoder = move_encoder or self.default_move_encoder
        self.max_samples = max_samples

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
        return len(self.line_offsets)
    
    def __getitem__(self, idx):
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
games_path = os.path.join(BASE_DIR, "data", "games.jsonl")
# Check if file exists and is non-empty before proceeding
if not os.path.isfile(games_path) or os.path.getsize(games_path) == 0:
    msg = f"❌ Dataset file not found or empty: {games_path}"
    print(msg)
    send_telegram_message(msg)
    print("✅ Telegram message sent.")
    sys.exit(1)
model = ChessNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Predeclare for potential loading
if os.path.exists(resume_checkpoint):
    print("🔄 Resuming from checkpoint...")
    checkpoint = torch.load(resume_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
else:
    print("🆕 Starting new training session.")
    start_epoch = 0
print("✅ Model initialized")
dataset = ChessPGNDataset(games_path, max_samples=1000000)
print(f"✅ Dataset instantiated: {len(dataset)} samples")
if len(dataset) == 0:
    msg = f"❌ Dataset loaded but contains 0 samples: {games_path}"
    print(msg)
    send_telegram_message(msg)
    print("✅ Telegram message sent.")
    sys.exit(1)

print("✅ DataLoader initialized")
                

def train_model(model, data, optimizer, start_epoch=0, epochs=2, batch_size=2048, device='cpu', pin_memory=False):
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    writer = SummaryWriter(log_dir=os.path.join(BASE_DIR, "runs", run_name))
    print(f"Logging to: runs/{run_name}")
    model.train()
    model.to(device)
    torch.backends.cudnn.benchmark = True
    all_losses = []
    all_rewards = []
    all_accuracies = []
    all_scores = []

    print("Starting training...")
    message = "✅ train.py started training..."
    print("⚠️ Attempting to send message:", message)
    send_telegram_message(message)
    print("✅ Telegram message sent.")
    print("✅ Starting epoch loop...")

    last_moves = None
    for epoch in range(start_epoch, epochs):
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item() if 'loss' in locals() else None,
            }, os.path.join(checkpoint_dir, "checkpoint_epoch_LAST.pth"))
            message = f"📦 train.py checkpoint saved — Epoch {epoch+1}"
            print("⚠️ Attempting to send message:", message)
            send_telegram_message(message)
            print("✅ Telegram message sent.")
        total_loss = 0
        total_reward = 0

        dataloader_iter = iter(dataloader)
        loss_policy = torch.tensor(0.0)
        loss_value = torch.tensor(0.0)
        preds_policy = torch.tensor([])
        preds_value = torch.tensor([])

        for i in range(len(dataloader)):
            try:
                boards_np, moves, outcomes = next(dataloader_iter)
                print(f"🔁 Processing batch {i+1}/{len(dataloader)}")
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
            rewards = outcomes  # Assuming outcome is reward signal
            total_reward += rewards.sum().item()
            writer.add_scalar("Metrics/Reward", rewards.sum().item(), epoch * len(dataloader) + i)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                preds_policy, preds_value = model(boards)

            if preds_policy.size(0) == moves.size(0):
                _, predicted_moves = torch.max(preds_policy, 1)
                batch_accuracy = (predicted_moves == moves).float().mean().item()
            else:
                batch_accuracy = 0.0

            loss_policy = F.cross_entropy(preds_policy.float(), moves)
            loss_value = F.mse_loss(preds_value.squeeze().float(), outcomes)
            loss = loss_policy + loss_value

            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Skipping batch due to invalid loss (NaN or Inf)")
                continue

            if i % 10 == 0:
                message = f"📦 Batch {i+1}/{len(dataloader)} — Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {batch_accuracy:.2%}"
                print("⚠️ Attempting to send message:", message)
                send_telegram_message(message)
                print("✅ Telegram message sent.")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isnan(loss):
                optimizer.step()

            for param_group in optimizer.param_groups:
                writer.add_scalar("Hyperparams/LearningRate", param_group["lr"], epoch)

            total_loss += loss.item()

            if i % 5 == 0:
                print(f"Epoch {epoch+1} | Batch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f} | GPU Mem: {torch.cuda.memory_allocated(device) / 1e6:.1f}MB")

        # ♟️ Self-play after each epoch
        print("♟️ Generating self-play games...")
        num_selfplay_games = int(os.getenv("NUM_SELFPLAY_GAMES", 50))
        new_selfplay_data = generate_self_play_data(model=model, num_games=num_selfplay_games, device=device)
        if new_selfplay_data:
            data.extend(new_selfplay_data)
            print(f"✅ {len(new_selfplay_data)} self-play games added to training set.")
        else:
            print("⚠️ No self-play games generated.")

        # 🔥 Log loss to TensorBoard (per epoch)
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
        writer.add_scalar("Metrics/AvgReward", total_reward / len(dataloader.dataset), epoch)

        avg_reward = total_reward / len(dataloader.dataset)
        score = (accuracy * 100) - (total_loss * 0.5) + (avg_reward * 10)
        score = max(0, min(score, 100))
        writer.add_scalar("Metrics/TrainingScore", score, epoch)
        print(f"\n🧠 Training Report — Epoch {epoch+1}")
        print(f"🎯 Accuracy: {accuracy * 100:.2f}%")
        print(f"📉 Loss: {total_loss:.4f}")
        print(f"🏋️‍♂️ Reward: {avg_reward:.4f}")
        print(f"📈 Score: {score:.2f}/100\n")

        if (epoch + 1) % 1 == 0:
            message = (
                f"📊 *Training Progress — Epoch {epoch+1}*\n"
                f"🎯 Accuracy: {accuracy * 100:.2f}%\n"
                f"📉 Loss: {total_loss:.4f}\n"
                f"🏋️‍♂️ Avg Reward: {avg_reward:.4f}\n"
                f"📈 Score: {score:.2f}/100"
            )
            print("⚠️ Attempting to send message:", message)
            send_telegram_message(message)
            print("✅ Telegram message sent.")

        if (epoch + 1) % 5 == 0:
            message = f"📊 train.py progress — Epoch {epoch+1}: Score {score:.2f}"
            print("⚠️ Attempting to send message:", message)
            send_telegram_message(message)
            print("✅ Telegram message sent.")

        all_losses.append(total_loss)
        all_rewards.append(total_reward / len(dataloader.dataset))
        all_accuracies.append(accuracy)
        all_scores.append(score)

        # Log model predictions and weights as histograms
        if preds_policy.numel() > 0:
            writer.add_histogram("Distributions/Policy", preds_policy, epoch)
        if preds_value.numel() > 0:
            writer.add_histogram("Distributions/Value", preds_value, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(f"Weights/{name}", param, epoch)

        logger.info(
            "Epoch %s/%s - Loss: %.4f - avg Reward: %.4f",
            epoch + 1,
            epochs,
            total_loss,
            total_reward / len(dataloader.dataset),
        )

    model.eval()

    writer.flush()
    writer.close()
    message = "🏁 *train.py finished training.*\nAll epochs completed successfully. Check TensorBoard for metrics and the checkpoints folder for saved models."
    print("⚠️ Attempting to send message:", message)
    send_telegram_message(message)
    print("✅ Telegram message sent.")
    return {
        "losses": all_losses,
        "rewards": all_rewards,
        "accuracies": all_accuracies,
        "scores": all_scores
    }

telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    global telegram
    if telegram_token and telegram_chat_id:
        try:
            if "telegram" not in globals():
                try:
                    import telegram
                except ImportError:
                    import subprocess
                    subprocess.run(["pip", "install", "python-telegram-bot==13.15"])
                    import telegram
            bot = telegram.Bot(token=telegram_token)
            bot.send_message(chat_id=telegram_chat_id, text=message, parse_mode=telegram.ParseMode.MARKDOWN)
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

# Send notification that training started
send_telegram_message("🚀 Training started...")
print("✅ Telegram message sent.")
print(f"✅ Telegram configured: TOKEN is {'set' if telegram_token else 'missing'}, CHAT_ID is {'set' if telegram_chat_id else 'missing'}")


# Function to capture stdout and stderr during training and send via Telegram in chunks
def capture_and_train():
    import io
    import contextlib
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            result = train_model(
                model=model,
                data=training_dataset,
                optimizer=optimizer,
                start_epoch=start_epoch,
                epochs=100,
                batch_size=2048,
                device=device,
                pin_memory=True
            )
    except Exception as e:
        msg = f"❌ Training failed: {e}"
        print(msg)
        send_telegram_message(msg)
        print("✅ Telegram message sent.")
        return
    output = buffer.getvalue()
    # Send a short summary first, then detailed output in chunks
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
    # Send output in chunks to avoid Telegram message size limits
    for i in range(0, len(output), 4000):
        chunk = output[i:i+4000]
        send_telegram_message(f"🧾 *Captured Training Output* (part {i//4000 + 1}):\n```{chunk}```")
        print("✅ Telegram message sent.")

# Main entry point for script execution
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    # Call the new function instead of direct train_model
    capture_and_train()
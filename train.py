print("Training script loaded...")
import os
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
BASE_DIR = os.getenv("BASE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
run_name = "chess_rl_v2"
checkpoint_dir = os.path.join(BASE_DIR, "runs", run_name, "checkpoints")
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_start_method("spawn", force=True)
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

model = ChessNet()
dataset = ChessPGNDataset(os.path.join(BASE_DIR, "data", "games.jsonl"), max_samples=1000000)
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True, collate_fn=custom_collate, pin_memory=False, num_workers=0)
                

def train_model(model, dataloader, epochs=10000, lr=1e-3):
    writer = SummaryWriter(log_dir=os.path.join(BASE_DIR, "runs", run_name))
    print(f"Logging to: runs/{run_name}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
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

    for epoch in range(epochs):
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
            message = f"📦 train.py checkpoint saved — Epoch {epoch+1}"
            print("⚠️ Attempting to send message:", message)
            send_telegram_message(message)
        total_loss = 0
        total_reward = 0
        for i, (boards_np, moves, outcomes) in enumerate(dataloader):
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
            message = f"📦 Batch {i+1}/{len(dataloader)} — Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {batch_accuracy:.2%}"
            print("⚠️ Attempting to send message:", message)
            send_telegram_message(message)

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

        # 🔥 Log loss to TensorBoard
        writer.add_scalar("Loss/Total", total_loss, epoch)
        writer.add_scalar("Loss/Policy", loss_policy.item(), epoch)
        writer.add_scalar("Loss/Value", loss_value.item(), epoch)

        if preds_policy.size(0) == moves.size(0):
            _, predicted_moves = torch.max(preds_policy, 1)
            accuracy = (predicted_moves == moves).float().mean().item()
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

        if (epoch + 1) % 5 == 0:
            message = f"📊 train.py progress — Epoch {epoch+1}: Score {score:.2f}"
            print("⚠️ Attempting to send message:", message)
            send_telegram_message(message)

        all_losses.append(total_loss)
        all_rewards.append(total_reward / len(dataloader.dataset))
        all_accuracies.append(accuracy)
        all_scores.append(score)

        # Log model predictions and weights as histograms
        writer.add_histogram("Distributions/Policy", preds_policy, epoch)
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

    writer.flush()
    writer.close()
    message = "🏁 train.py finished training."
    print("⚠️ Attempting to send message:", message)
    send_telegram_message(message)
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
            bot.send_message(chat_id=telegram_chat_id, text=message)
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

# Send notification that training started
send_telegram_message("🚀 Training started...")
print(f"✅ Telegram message function test: {telegram_token=}, {telegram_chat_id=}")

train_model(model, dataloader, epochs=10000, lr=1e-3)
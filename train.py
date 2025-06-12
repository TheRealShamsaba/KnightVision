import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def train_model(model, data, epochs=100, batch_size=32, lr=1e-3):
    writer = SummaryWriter(log_dir=f"runs/chess_rl_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)
    all_losses = []
    all_rewards = []
    all_accuracies = []

    for epoch in range(epochs):
        random.shuffle(data)
        total_loss = 0
        total_reward = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            boards = torch.tensor([item[0] for item in batch]).float()
            moves = torch.tensor([item[1] for item in batch]).long()
            outcomes = torch.tensor([item[2] for item in batch]).float()

            boards = boards.to(device)
            moves = moves.to(device)
            outcomes = outcomes.to(device)
            rewards = outcomes  # Assuming outcome is reward signal
            total_reward += rewards.sum().item()
            writer.add_scalar("Metrics/Reward", rewards.sum().item(), epoch * len(data) + i)

            preds_policy, preds_value = model(boards)
            loss_policy = F.cross_entropy(preds_policy, moves)
            loss_value = F.mse_loss(preds_value.squeeze(), outcomes)
            loss = loss_policy + loss_value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for param_group in optimizer.param_groups:
                writer.add_scalar("Hyperparams/LearningRate", param_group["lr"], epoch)

            total_loss += loss.item()

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
        writer.add_scalar("Metrics/AvgReward", total_reward / len(data), epoch)

        all_losses.append(total_loss)
        all_rewards.append(total_reward / len(data))
        all_accuracies.append(accuracy)

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
            total_reward / len(data),
        )

    writer.close()
    return {
        "losses": all_losses,
        "rewards": all_rewards,
        "accuracies": all_accuracies
    }
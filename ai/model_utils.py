import os
import torch
import torch.optim as optim
from ai.model import ChessNet

# Default checkpoint path and device setup
resume_checkpoint = "checkpoints/latest.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_or_initialize_model(model_class, optimizer_class, optimizer_kwargs, model_path, device):
    model = model_class().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    start_epoch = 0
    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint and "optimizer_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            print("🔁 Checkpoint loaded.")
        else:
            print("⚠️ Checkpoint is missing expected keys. Initializing fresh model.")
    else:
        print("🆕 Initializing new model and optimizer.")
    if torch.cuda.device_count() > 1:
        print(f"🌐 Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)
    return model, optimizer, start_epoch
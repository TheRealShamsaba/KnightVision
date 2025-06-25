import os
import torch
import torch.optim as optim
from model import ChessNet

# Default checkpoint path and device setup
resume_checkpoint = "checkpoints/latest.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_or_initialize_model(model_class, optimizer_class, optimizer_kwargs, checkpoint_path, device):
    model = model_class().to(device)
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint and "optimizer_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            print("🔁 Checkpoint loaded.")
        else:
            print("⚠️ Checkpoint is missing expected keys. Initializing fresh model.")
    else:
        print("🆕 Initializing new model and optimizer.")
    
    return model, optimizer, start_epoch

# --- Model & optimizer loading ---
model, optimizer, start_epoch = load_or_initialize_model(
    model_class=ChessNet,
    optimizer_class=optim.Adam,
    optimizer_kwargs={"lr": 1e-3},
    checkpoint_path=resume_checkpoint,
    device=device
)
# If multiple GPUs available, wrap model in DataParallel
if torch.cuda.device_count() > 1:
    logger.info(f"🌐 Using DataParallel on {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
print(f"✅ Model initialized. Resuming at epoch {start_epoch}")
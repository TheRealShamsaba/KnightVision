from model_utils import load_or_initialize_model
import os
import torch
import torch.optim as optim
from model import ChessNet

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

 # basicChess
+
+This project contains a simple chess game and a reinforcement learning setup for training a model to play.
+
+## Dependencies
+
+- Python 3.11+
+- pygame
+- torch
+- numpy
+- tensorboard
+
+Install them with:
+
+```bash
+pip install pygame torch numpy tensorboard
+```
+
+## Running the Game
+
+Launch the graphical game interface with:
+
+```bash
+python chessMain.py
+```
+
+Click pieces to move them. Press `Z` to undo the last move.
+
+## Training the Model
+
+The reinforcement loop that generates self‑play games and trains the network is in `learn.py`:
+
+```bash
+python learn.py
+```
+
+It saves the trained weights to `model.pth` and logs metrics under `runs/`. View logs using TensorBoard:
+
+```bash
+tensorboard --logdir runs
+```
+
+You can also run the components individually:
+
+```bash
+python self_play.py   # create training data
+python train.py       # train using existing data
+```
+
+## Files
+
+- `chessMain.py` – pygame interface for playing chess
+- `chessEngine.py` – game state and move validation
+- `model.py` – PyTorch neural network
+- `self_play.py` – generates data via self-play
+- `train.py` – training utilities
+- `learn.py` – orchestrates self-play and training

## Telegram Notifications

Training utilities can optionally send progress updates through a Telegram bot. Configure these variables in your environment or `.env` file:

- `TELEGRAM_BOT_TOKEN` – bot API token
- `TELEGRAM_CHAT_ID` – chat ID that receives notifications
- `TELEGRAM_ENABLED` – set to `0` to disable messages (defaults to `1`)
- `TELEGRAM_NOTIFY_INTERVAL` – minimum seconds between messages

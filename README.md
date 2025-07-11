# ♟️ KnightVision — Reinforcement Learning-based Chess Engine

KnightVision is an advanced chess engine combining classic move-generation logic with modern deep learning and reinforcement learning. Built to learn and adapt, it continuously improves through self-play and data-driven training.
📄 **Thesis**: [📘 Read the full graduation report](https://drive.google.com/file/d/1WmA_4Vbek0yzG7596a_SsLYBHnRwpR9f/view?usp=sharing)
---

## 🚀 Overview

* 🧠 **Custom Neural Network** with policy and value heads
* ♻️ **Reinforcement Learning** via self-play and human PGN data
* 📈 **TensorBoard integration** for tracking progress
* 🗂️ **PGN parsing pipeline** for large-scale datasets

---

## ✨ Features

✅ Fully custom neural architecture
✅ Self-play loop with reinforcement updates
✅ PGN parsing and JSONL dataset support
✅ Telegram live reporting
✅ Flexible configuration and experiments

---

## 📁 Project Structure

```bash
├── ai/              # Neural network models, utilities, logging
├── bot/             # Telegram bot and notification utilities
├── core/            # Core chess logic (engine and main game logic)
├── data_utils/      # PGN parsing and dataset utilities
├── scripts/         # Training, self-play, evaluation entry-point scripts
├── checkpoints/     # Saved model weights
├── runs/            # TensorBoard logs
├── images/          # Supporting images
├── notebooks/       # Analysis and experimentation notebooks
├── requirements.txt # Dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/KnightVision.git
cd KnightVision
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
python train.py        # Start supervised or initial training
python learn.py        # Run full reinforcement learning loop
python play_vs_model.py  # Play against the engine
```

---

## 🗺️ Roadmap

### ✅ Completed

* PGN parsing & dataset integration
* Basic move legality and board logic
* Initial supervised training
* Self-play reinforcement learning

### 🚧 In Progress

* Advanced pruning & evaluation techniques
* Web UI for online play and analysis
* Further architecture tuning & experiments

### 💡 Future Ideas

* Real-time online ladder system
* Adaptive style learning against different opponents
* Detailed game analysis and commentary features

---

## 🤝 Contributing

Pull requests and issues are welcome! Let’s build better chess intelligence together.

---

## 🧑‍💻 Author

https://github.com/TheRealShamsaba
https://github.com/BakerDmo

---

## ⭐️ Support

If you like this project, give it a ⭐️ to help it grow!

---

## 📄 License

MIT License

# ♟️ KnightVision — Reinforcement Learning-based Chess Engine

KnightVision is a cutting-edge chess engine that combines classic move generation logic with modern deep learning and reinforcement learning. Built to learn and adapt, it continuously improves through self-play and data-driven training.

📄 **Thesis**: [📘 Read the full graduation report](https://drive.google.com/file/d/1WmA_4Vbek0yzG7596a_SsLYBHnRwpR9f/view?usp=sharing)

---

## 🚀 Overview

- 🧠 **Custom Neural Network** with policy and value heads
- ♻️ **Reinforcement Learning** via self-play and human PGN data
- 📈 **TensorBoard integration** for monitoring and visualization
- 🗂️ **PGN parsing pipeline** for large-scale datasets
- 🤖 **Telegram bot integration** for live notifications

---

## ✨ Features

✅ Fully custom neural architecture  
✅ Continuous improvement through self-play  
✅ Large-scale PGN parsing and batch dataset support  
✅ Real-time Telegram reporting during training and self-play  
✅ Modular, clean project structure  

---

## 📁 Project Structure

```bash
├── ai/              # Neural network models, utilities, logging
├── bot/             # Telegram bot and notification utilities
├── core/            # Core chess logic (engine and game logic)
├── data_utils/      # PGN parsing and dataset utilities
├── scripts/         # Training, self-play, evaluation entry points
├── checkpoints/     # Saved model weights
├── runs/            # TensorBoard logs
├── images/          # Supporting images or diagrams
├── notebooks/       # Analysis and experimentation notebooks
├── requirements.txt # Dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/KnightVision.git
cd KnightVision
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage

### 🎯 Train (supervised or initial training)

```bash
PYTHONPATH=. python scripts/train.py
```

### ♻️ Full reinforcement learning loop

```bash
PYTHONPATH=. python scripts/learn.py
```

### 🤖 Self-play data generation

```bash
PYTHONPATH=. python scripts/self_play.py
```

### ♟️ Play against the engine

```bash
PYTHONPATH=. python scripts/play_vs_model.py
```

### 🔎 Evaluate against Stockfish

```bash
PYTHONPATH=. python scripts/stockfish_play.py
```

---

## 🗺️ Roadmap

### ✅ Completed

- PGN parsing & dataset integration
- Basic move legality and board logic
- Initial supervised training
- Self-play reinforcement learning

### 🚧 In Progress

- Advanced pruning & evaluation techniques
- Web UI for online play and analysis
- Further architecture tuning & experiments

### 💡 Future Ideas

- Real-time online ladder system
- Adaptive style learning vs. different opponents
- Advanced game analysis and commentary

---

## 🤝 Contributing

Pull requests and issues are welcome! Let's build the next generation of chess intelligence together.

---

## 🧑‍💻 Authors

- [TheRealShamsaba](https://github.com/TheRealShamsaba)
- [BakerDmo](https://github.com/BakerDmo)

---

## ⭐️ Support

If you like this project, please give it a ⭐️ to help it grow!

---

## 📄 License

MIT License
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

├── basicChess/        # Core engine, model, and training scripts
├── data/             # Datasets and checkpoints
├── runs/             # TensorBoard logs
├── requirements.txt  # Dependencies
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

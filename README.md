# KnightVision

A chess-engine project combining custom board and move logic with PyTorch policy/value networks, PGN data processing, and self-play experiments.

Built by [Arian Shamsabadi](https://github.com/TheRealShamsaba) and [BakerDmo](https://github.com/BakerDmo).

## What's here

- A custom chess rules engine with move generation, make/undo operations, castling, en passant, and promotion tests.
- A residual neural network with separate move-policy and position-value heads.
- Scripts for supervised training, self-play, and evaluation against Stockfish.
- A Pygame board interface, PGN/JSONL data utilities, and Telegram reporting utilities.

**Status:** an experimental engineering project. The rules tests can run independently of the machine-learning stack. Training and model play require additional environment, data, and path setup; this repository does not provide a ready-to-run trained engine or a verified playing-strength benchmark.

## Quick start: run the rules tests

Use Python 3.11. No third-party packages, model weights, network access, or credentials are needed for these tests.

```bash
git clone https://github.com/TheRealShamsaba/KnightVision.git
cd KnightVision
PYTHONPATH=core python3 -B -m unittest discover -s tests -v
```

Run the test command from the repository root. It discovers eight tests covering castling, en passant capture/undo, and pawn promotion/undo. `PYTHONPATH=core` resolves the tests' existing `chessEngine` import directly, avoiding the Pygame import in `core/__init__.py`. These tests cover specific rules scenarios, not every legal position or neural-network behavior.

## Repository map

| Location | Purpose |
| --- | --- |
| [`core/chessEngine.py`](core/chessEngine.py) | Board state, moves, and rules |
| [`core/chessMain.py`](core/chessMain.py) | Pygame board interface |
| [`ai/model.py`](ai/model.py) | Residual network with policy and value heads |
| [`ai/`](ai/) | Encoding, model loading, and logging utilities |
| [`data_utils/`](data_utils/) | PGN parsing and dataset preparation |
| [`scripts/`](scripts/) | Training, self-play, model play, and Stockfish evaluation |
| [`tests/`](tests/) | Rules regression tests |
| [`images/`](images/) | Chess-piece sprites used by Pygame |
| [`bot/`](bot/) | Telegram reporting and subscription utilities |

## Working with the experiments

Start by reading the relevant entry point and its assumptions:

| Entry point | Current setup requirements |
| --- | --- |
| [`scripts/train.py`](scripts/train.py) | PGN-derived JSONL data; imports Google Colab and uses `/content/drive/MyDrive/KnightVision` by default. Some paths can be set through environment variables. |
| [`scripts/learn.py`](scripts/learn.py) | Orchestrates training, self-play, and evaluation; contains Colab-specific paths and reporting hooks. |
| [`scripts/self_play.py`](scripts/self_play.py) | Self-play functions requiring a compatible model checkpoint; uses multiprocessing. |
| [`scripts/play_vs_model.py`](scripts/play_vs_model.py) | Pygame, piece images, and a named checkpoint. Its local `BASE_DIR` points to `scripts/`, while the images live at the repository root; paths need adjustment before use. |
| [`scripts/stockfish_play.py`](scripts/stockfish_play.py) | A model checkpoint and separately installed Stockfish binary. The CLI requires `--model-path`; its default Stockfish path is `/usr/games/stockfish`. |

The historical dependency lists in [`requirements.txt`](requirements.txt) and [`setup.py`](setup.py) disagree on PyTorch, torchvision, TensorBoard, and TensorFlow versions. They also omit imports used by some scripts, including `psutil`, `tqdm`, `google.colab`, and `python-telegram-bot`. A complete training environment has not been validated here. Resolve these differences for the intended runtime before installing the ML stack or starting a long run.

Training datasets and `.pth` checkpoints are not included in the current tracked files. Training can be expensive, and some entry points load environment files, write outputs, or contact Telegram. The lightweight test command above does none of those things. Training, self-play, the GUI, and Stockfish evaluation were not run as part of the documentation checks.

## Known limits

- Passing the eight rules tests does not establish complete rules correctness, an Elo rating, or improved play through training.
- The move policy has 4,096 from/to-square outputs; it does not represent each underpromotion choice separately.
- Colab paths, checkpoint formats, and GUI asset paths need reconciliation before the experiment scripts can be treated as portable commands.
- No hosted demo or web interface is included.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local workflow and [LICENSE](LICENSE) for the MIT license. Please retain both authors' credit when building on the project.

[Arian's profile](https://github.com/TheRealShamsaba) · [Portfolio](https://github.com/TheRealShamsaba/Portfolio) · [Movie Recommender System](https://github.com/TheRealShamsaba/Movie-Recommender-system)

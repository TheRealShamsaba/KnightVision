# KnightVision v2

## What was wrong in v1

Training (`scripts/train.py`) and everything that used the model afterwards (play, self-play, Stockfish evaluation) encoded positions differently:

| | Training | Play / self-play / evaluation (v1) |
|---|---|---|
| Piece planes | P N B R Q K, then p n b r q k | K Q R B N P, then k q r b n p |
| Move index | python-chess squares, a1 = 0 | rows counted from rank 8, a8 = 0 |

So at play time the network read kings as pawns and queens as knights, and its move probabilities were read off a vertically mirrored board. Self-play then trained on that second layout, mixing two board layouts into one network.

Two more bugs sat next to it:

- Self-play batched positions for inference but chose each move from the output of an older position (often from an earlier game), so self-play moves were mostly noise.
- `stockfish_play.py` pushed pawn moves to the last rank without a promotion piece.

The custom rules engine (`core/chessEngine.py`, used by self-play and the Pygame board) also disagreed with python-chess. The new test found these by comparing the two over random games:

- Knight checks from one of the eight knight jumps were never detected.
- Attacks were worked out by generating the opponent's moves, so pawn pushes counted as attacks, pawn captures onto empty squares didn't, and kings could step into a pawn's capture square.
- A pawn pinned along its own file couldn't push, because the pin direction's sign was compared the wrong way round.
- En passant couldn't capture a checking pawn, and wasn't checked for exposing the king along the rank.

After the fixes, the engine and python-chess agree at every ply of 2,500 random games.

### Same checkpoint, before and after

`Best_Model.pth`'s top four moves, with the share of the network's probability on each:

| Position | v1 encoding | v2 encoding |
|---|---|---|
| Start | g3 25%, e3 12%, d3 11%, f3 9% | e4 61%, d4 23%, Nf3 4%, c4 3% |
| 1.e4 | f6 48%, g6 14%, f5 11%, d6 5% | e5 33%, d5 18%, c5 17%, e6 9% |
| 1.e4 e5 | Ke2 22%, g3 19%, f3 10%, d4 5% | Nf3 68%, Bc4 8%, f4 6%, Nc3 6% |
| 1.e4 e5 2.Nf3 | Be7 16%, g6 13%, f6 11%, f5 8% | Nc6 53%, d6 22%, Nf6 13%, d5 4% |
| 1.d4 d5 2.c4 | g6 20%, f6 15%, e6 14%, b6 11% | c6 27%, e6 27%, Nf6 22%, dxc4 11% |
| 1.e4 c5 2.Nf3 d6 | Rg1 26%, g3 19%, Ke2 15%, Ng5 5% | Bc4 28%, d4 27%, c3 14%, Nc3 10% |

The network learned openings; v1 just never showed it the board it learned on.

## The fix (done)

- `ai/ai.py` is the single encoder, frozen to the layout the checkpoints were trained on. Training, the dataset loader, self-play, play and Stockfish evaluation all import it.
- `tests/test_encoding.py` compares it against a frozen copy of the original training encoder, and plays random games in both the custom rules engine and python-chess to check board tensors and legal-move indices agree at every ply.
- Self-play evaluates the current position on every move.
- Stockfish evaluation maps legal moves through `legal_move_indices`, promoting to a queen.
- The rules engine fixes above, each with a fixed position in `tests/test_encoding.py`.

Run the tests with numpy, python-chess and torch installed (without them the encoding tests skip):

```bash
PYTHONPATH=core python3 -B -m unittest discover -s tests -v
```

The browser engine in the portfolio (`prtofolio/src/next/kvEngine.ts`) already uses this same layout, so the exported ONNX model and the Python code agree.

## Phase 1: measure it

The portfolio story needs a number.

1. Install Stockfish (`brew install stockfish`).
2. Play `Best_Model.pth` against Stockfish at skill levels 0, 1, 3 and 5, 40 games each with colours alternating, once with the v1 encoding (`git stash` the fix) and once with v2.
3. Report score per level, and an estimated Elo from the skill-level ratings.
4. If PGN data is reachable (Drive), report top-1 move accuracy on held-out positions with both encodings.

## Phase 2: the web interface

**Built as a 3D flight instead, in `web/`** (see `web/README.md`). The plan below was the first idea.

A static site, deployable on Vercel next to the portfolio and linked from its KnightVision card. No server: the model runs in the browser with ONNX Runtime Web, reusing the portfolio's export (`knightvision.onnx`, 25 MB int8) and `kvEngine.ts`.

Stack: Vite, React, TypeScript, Tailwind (same as the portfolio), chess.js for rules, ONNX Runtime Web (WASM, in its own worker).

### Screens

**Play**
- Board with drag and click moves, choose a colour, undo, flip, new game.
- Side panel: the network's top five candidate moves with probabilities, drawn as arrows on the board (thickness follows probability).
- Move list in SAN; export PGN.
- Strength switch: "Instinct" (pure policy, one forward pass) and later "Think" (search, phase 4).

**Engine's eyes**
- Heatmap of where the network wants to move to, summed over destination squares.
- Toggle the 12 input planes to show exactly what the network is given.

**How it works**
- Architecture diagram: 12×8×8 input, two convolutions, five residual blocks of 512 channels, a 4096-way policy head and a value head.
- Training pipeline: Lichess PGN, then supervised learning, then self-play, then Stockfish evaluation.
- The bug story, with the phase 1 before/after numbers.

### Keeping the encoders in sync

The v1 bug was two encoders drifting apart. The browser adds a third, so:

- A Python script writes `fixtures/encoding.json`: 200 positions with their FEN, encoded planes and legal move indices.
- A Vitest test checks `kvEngine.ts` against that file, so a mismatch fails CI instead of quietly weakening the engine.
- An ONNX parity check runs PyTorch and the ONNX model on the same positions and compares the top move and the logits.

## Phase 3: polish and ship

- Loading state for the 25 MB model (progress bar, cached by the browser after the first visit).
- Mobile layout: board on top, candidates below.
- Light and dark themes matching the portfolio.
- Shareable positions via FEN in the URL hash.

## Phase 4: engine improvements (retraining)

These change the input or the targets, so they need a new training run and a new export:

- Add planes for side to move, castling rights and en passant. Without them the network can't tell whose turn it is.
- Train the value head from the side to move's point of view. The current checkpoint's value output isn't usable.
- Add a small search (MCTS or alpha-beta over the policy and value heads) for the "Think" mode.
- Re-run self-play on the fixed code before trusting any reinforcement learning results.

## Art: the policy quilt

`scripts/policy_quilt.py` draws a whole game as a print: one tile per position, each showing all 4,096 move probabilities from a single forward pass. A tile is a board of boards: the large square is where a piece starts, the small board inside it is where the network would send it. The played move is ringed and captioned with its share among legal moves.

```bash
PYTHONPATH=. python scripts/policy_quilt.py --model-path Best_Model.pth --game opera
PYTHONPATH=. python scripts/policy_quilt.py --model-path Best_Model.pth --game kv-win --theme paper
PYTHONPATH=. python scripts/policy_quilt.py --model-path Best_Model.pth --pgn mygame.pgn --paper A1
```

It writes a PNG (sRGB, 300 dpi, 3 mm bleed by default), a PDF and a JPEG preview to `runs/quilt/`. Presets: `opera` (Morphy, Paris 1858) and `kv-win` (KnightVision's 17.Nc7# win over Stockfish at 1320). Paper sizes: A4 to A1, 18x24 and 24x36 inches. Themes: `night` and `paper`.

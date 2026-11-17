"""
Exports the network's real outputs that the website draws from, so its story sections need
no model download. Live play on the site runs the ONNX model in the browser instead.

    PYTHONPATH=. python scripts/export_web_data.py --model-path Best_Model.pth \
        --elo-games runs/elo/v2_best_model/games.jsonl

Writes web/src/data/knightvision.json.
"""
import argparse
import base64
import json
import os

import chess
import numpy as np
import torch

from ai import encode_board, legal_move_indices
from ai.model import ChessNet

OPERA = ("e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 "
         "Bxb5+ Nbd7 O-O-O Rd8 Rxd7 Rxd7 Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#").split()

KV_WIN = ("e4 Nc6 Nf3 Nf6 e5 Nd5 d4 d6 c4 dxe5 cxd5 e4 dxc6 exf3 cxb7 fxg2 Bxg2 Rb8 Bc6+ Bd7 "
          "Bxd7+ Qxd7 O-O c6 Nc3 g6 Be3 c5 dxc5 Qh3 Nd5 Rd8 Nc7#").split()

MIRROR_LINES = ["", "e4", "e4 e5", "e4 e5 Nf3", "d4 d5 c4", "e4 c5 Nf3 d6"]

# The position before Morphy's 10.Nxb5, used for the constellation.
CONSTELLATION_PLY = 18
# The position before 6.Bc4, used to show what flows through the layers.
LAYERS_PLY = 10
# A position from ordinary human play, used to show what training changes.
LEARNING_LINE = "e4 e5"
LEARNING_ANSWER = "Nf3"
# From scripts/train.py and the checkpoint.
TRAINING = {"positions": 5_000_000, "validation": 0.1, "batch": 4096, "optimizer": "Adam", "lr": 5e-4}
LOG_FLOOR = -7


def policy(model, planes):
    with torch.no_grad():
        logits = model(torch.from_numpy(planes).unsqueeze(0))[0][0].numpy().astype(np.float64)
    p = np.exp(logits - logits.max())
    return p / p.sum()


def legal_shares(board, p):
    legal = legal_move_indices(board)
    indices = list(legal)
    shares = p[indices] / p[indices].sum()
    order = np.argsort(-shares)
    return [(legal[indices[i]], float(shares[i])) for i in order], float(p[indices].sum())


def v1_planes(board):
    """How v1 fed the network: planes in K Q R B N P order instead of P N B R Q K."""
    planes = np.zeros((12, 8, 8), np.float32)
    for square, piece in board.piece_map().items():
        plane = "KQRBNP".index(piece.symbol().upper()) + (0 if piece.color else 6)
        planes[plane, 7 - chess.square_rank(square), chess.square_file(square)] = 1
    return planes


def v1_index(square):
    """v1 numbered move squares from rank 8 down, so it read every move upside down."""
    return (7 - chess.square_rank(square)) * 8 + chess.square_file(square)


def quantise(p):
    """4,096 probabilities as bytes on a log scale: 0 is one in ten million or less, 255 is certain."""
    v = np.clip((np.log10(np.maximum(p, 1e-12)) - LOG_FLOOR) / -LOG_FLOOR, 0, 1)
    return base64.b64encode((v * 255).round().astype(np.uint8).tobytes()).decode()


def game_record(model, sans, with_quilt):
    board, plies = chess.Board(), []
    for san in sans:
        p = policy(model, encode_board(board))
        shares, legal_mass = legal_shares(board, p)
        move = board.parse_san(san)
        share = dict((m, s) for m, s in shares).get(move, 0.0)
        ply = {
            "fen": board.fen(), "san": san, "uci": move.uci(), "share": round(share, 4),
            "legalMass": round(legal_mass, 4),
            "top": [{"san": board.san(m), "uci": m.uci(), "p": round(s, 4)} for m, s in shares[:6]],
        }
        if with_quilt:
            ply["quilt"] = quantise(p)
        plies.append(ply)
        board.push(move)
    return {"plies": plies, "finalFen": board.fen(), "result": board.result()}


def mirror(model):
    rows = []
    for line in MIRROR_LINES:
        board = chess.Board()
        for san in line.split():
            board.push_san(san)
        v2, _ = legal_shares(board, policy(model, encode_board(board)))
        p1 = policy(model, v1_planes(board))
        legal = [m for m in board.legal_moves if m.promotion in (None, chess.QUEEN)]
        scores = np.array([p1[v1_index(m.from_square) * 64 + v1_index(m.to_square)] for m in legal])
        scores /= scores.sum()
        v1 = sorted(zip(legal, scores), key=lambda t: -t[1])
        rows.append({
            "line": line, "fen": board.fen(),
            "v1": [{"san": board.san(m), "uci": m.uci(), "p": round(float(s), 4)} for m, s in v1[:4]],
            "v2": [{"san": board.san(m), "uci": m.uci(), "p": round(s, 4)} for m, s in v2[:4]],
        })
    return rows


def constellation(model):
    board = chess.Board()
    for san in OPERA[:CONSTELLATION_PLY]:
        board.push_san(san)
    p = policy(model, encode_board(board))
    legal = set(legal_move_indices(board))
    top = np.argsort(-p)[:48]
    return {
        "fen": board.fen(), "next": OPERA[CONSTELLATION_PLY],
        "legalMass": round(float(p[list(legal)].sum()), 4),
        "beliefs": [{"from": chess.square_name(int(i) // 64), "to": chess.square_name(int(i) % 64),
                     "p": round(float(p[i]), 5), "legal": int(i) in legal} for i in top],
    }


def layers(model):
    """What flows through the network for one position: the six liveliest 8×8 maps from each stage."""
    board = chess.Board()
    for san in OPERA[:LAYERS_PLY]:
        board.push_san(san)
    stages = [("Opening convolution", model.conv1), ("Widening convolution", model.conv2)]
    stages += [(f"Residual block {i + 1}", block) for i, block in enumerate(model.res_blocks)]
    stages += [("Policy head", model.policy_conv)]
    # Every weight belongs to one stage, batch norms and the policy head's final layer included.
    owned = {model.conv1: [model.conv1, model.bn1], model.conv2: [model.conv2, model.bn2],
             model.policy_conv: [model.policy_conv, model.policy_bn, model.policy_fc]}
    captured = {}
    hooks = [module.register_forward_hook(lambda m, i, o, name=name: captured.__setitem__(name, o[0].detach()))
             for name, module in stages]
    with torch.no_grad():
        model(torch.from_numpy(encode_board(board)).unsqueeze(0))
    for h in hooks:
        h.remove()
    out = []
    for name, module in stages:
        act = captured[name]
        if module is model.conv1 or module is model.conv2:
            act = torch.relu(act)  # the network applies batch norm and ReLU next; show the signal that survives
        spread = act.flatten(1).std(1)
        maps = []
        for c in torch.argsort(spread, descending=True)[:6]:
            m = act[c].numpy()
            m = (m - m.min()) / (m.max() - m.min() + 1e-9)
            maps.append(base64.b64encode((m * 255).round().astype(np.uint8).tobytes()).decode())
        out.append({"name": name, "channels": int(act.shape[0]), "maps": maps,
                    "parameters": sum(p.numel() for m in owned.get(module, [module]) for p in m.parameters())})
    return {"fen": board.fen(), "stages": out}


def filters(model, count=4):
    """
    The first layer's 3×3 filters for its busiest channels on the Opera position, with the batch
    norm that follows folded in, so out = relu(sum(w · patch) + b) is exactly what the network computes.
    """
    board = chess.Board()
    for san in OPERA[:LAYERS_PLY]:
        board.push_san(san)
    x = torch.from_numpy(encode_board(board)).unsqueeze(0)
    bn = model.bn1
    with torch.no_grad():
        act = torch.relu(bn(model.conv1(x)))[0]
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = model.conv1.weight * scale[:, None, None, None]
        b = (model.conv1.bias - bn.running_mean) * scale + bn.bias
    picked = torch.argsort(act.flatten(1).std(1), descending=True)[:count]
    return [{"channel": int(c), "weights": [round(float(v), 4) for v in w[c].flatten()], "bias": round(float(b[c]), 4)}
            for c in picked]


def learning(model):
    """The same position through an untrained network (random weights) and the trained one."""
    board = chess.Board()
    for san in LEARNING_LINE.split():
        board.push_san(san)
    answer = board.parse_san(LEARNING_ANSWER)
    torch.manual_seed(0)
    untrained = ChessNet().eval()
    out = {"fen": board.fen(), "answer": LEARNING_ANSWER, "answerUci": answer.uci(), "training": TRAINING}
    for name, net in (("untrained", untrained), ("trained", model)):
        p = policy(net, encode_board(board))
        shares, _ = legal_shares(board, p)
        out[name] = {
            "top": [{"san": board.san(m), "uci": m.uci(), "p": round(s, 4)} for m, s in shares[:6]],
            "answerShare": round(dict(shares)[answer], 4),
            # Training scored the full 4,096-way guess: loss = −ln p(the move the human played).
            "loss": round(float(-np.log(p[answer.from_square * 64 + answer.to_square])), 3),
            "legalMoves": len(shares),
        }
    return out


def elo(games_path):
    if not games_path or not os.path.exists(games_path):
        return None
    games = []
    with open(games_path) as f:
        for line in f:
            g = json.loads(line)
            games.append({"elo": g["elo"], "white": g["kv_white"], "score": g["kv_score"], "plies": g["plies"]})
    summary_path = os.path.join(os.path.dirname(games_path), "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    return {"rating": summary.get("rating"), "margin": summary.get("margin95"), "games": games}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--elo-games", default=os.path.join("runs", "elo", "v2_best_model", "games.jsonl"))
    parser.add_argument("--out", default=os.path.join("web", "src", "data", "knightvision.json"))
    args = parser.parse_args()

    state = torch.load(args.model_path, map_location="cpu")
    model = ChessNet()
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()

    data = {
        "checkpoint": {"epoch": state.get("epoch"), "validationLoss": round(float(state["loss"]), 3) if "loss" in state else None},
        "parameters": sum(p.numel() for p in model.parameters()),
        "opera": game_record(model, OPERA, with_quilt=True),
        "win": game_record(model, KV_WIN, with_quilt=False),
        "mirror": mirror(model),
        "constellation": constellation(model),
        "layers": layers(model),
        "learning": learning(model),
        "filters": filters(model),
        "elo": elo(args.elo_games),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"{args.out}  {os.path.getsize(args.out) / 1024:.0f} KB")


if __name__ == "__main__":
    main()

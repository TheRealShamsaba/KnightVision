"""
Renders a policy quilt: a print-ready poster of one game, with one tile per position showing
all 4,096 from→to probabilities KnightVision gives that position in a single forward pass.

Each tile is a board of boards. The large 8×8 squares are where a piece moves from; the small
8×8 board inside each one is where the network would send it. The move that was played is ringed.

    PYTHONPATH=. python scripts/policy_quilt.py --model-path Best_Model.pth --game opera
    PYTHONPATH=. python scripts/policy_quilt.py --model-path Best_Model.pth --pgn game.pgn --paper A1 --theme paper

Writes a PNG (sRGB, DPI set), a PDF and a small JPEG preview to --out.
"""
import argparse
import math
import os
from datetime import date

import chess
import chess.pgn
import numpy as np
import torch
from PIL import Image, ImageCms, ImageDraw, ImageFont

from ai import encode_board, legal_move_indices
from ai.model import ChessNet

PAPERS_MM = {"A4": (210, 297), "A3": (297, 420), "A2": (420, 594), "A1": (594, 841),
             "18x24": (457.2, 609.6), "24x36": (609.6, 914.4)}

GAMES = {
    "opera": {
        "title": "The Opera Game",
        "subtitle": "Paul Morphy vs. Duke Karl of Brunswick and Count Isouard · Paris, 1858",
        "moves": "e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 "
                 "Bxb5+ Nbd7 O-O-O Rd8 Rxd7 Rxd7 Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#",
    },
    "kv-win": {
        "title": "KnightVision vs. Stockfish",
        "subtitle": "KnightVision with White, no search · Stockfish limited to 1320 Elo, 60 s + 0.6 s",
        "moves": "e4 Nc6 Nf3 Nf6 e5 Nd5 d4 d6 c4 dxe5 cxd5 e4 dxc6 exf3 cxb7 fxg2 Bxg2 Rb8 Bc6+ Bd7 "
                 "Bxd7+ Qxd7 O-O c6 Nc3 g6 Be3 c5 dxc5 Qh3 Nd5 Rd8 Nc7#",
    },
}

# Colour stops run from "never" (1 in 10 million or less) to "certain", on a log scale.
THEMES = {
    "night": {
        "bg": (12, 13, 18), "fg": (255, 243, 230), "muted": (128, 134, 158), "accent": (226, 92, 63),
        "stops": [(22, 24, 33), (42, 54, 96), (201, 69, 47), (242, 193, 78), (255, 243, 230)],
    },
    "paper": {
        "bg": (244, 238, 226), "fg": (24, 21, 24), "muted": (124, 113, 102), "accent": (184, 56, 36),
        "stops": [(235, 228, 213), (196, 182, 160), (201, 69, 47), (112, 30, 26), (24, 21, 24)],
    },
}
LOG_FLOOR = -7

FONT_FILES = {
    "display": [("/System/Library/Fonts/Supplemental/Didot.ttc", 0), ("DejaVuSerif.ttf", 0)],
    "italic": [("/System/Library/Fonts/Supplemental/Didot.ttc", 1), ("DejaVuSerif-Italic.ttf", 0)],
    "text": [("/System/Library/Fonts/Avenir Next.ttc", 7), ("DejaVuSans.ttf", 0)],
    "mono": [("/System/Library/Fonts/Menlo.ttc", 0), ("DejaVuSansMono.ttf", 0)],
}


def font(role, size):
    for path, index in FONT_FILES[role]:
        try:
            return ImageFont.truetype(path, max(1, int(size)), index=index)
        except OSError:
            continue
    return ImageFont.load_default(size=max(1, int(size)))


def load_game(args):
    if args.pgn:
        with open(args.pgn) as f:
            for _ in range(args.game_index + 1):
                game = chess.pgn.read_game(f)
        h = game.headers
        details = [h.get(k) for k in ("Event", "Site", "Date") if h.get(k, "?") not in ("?", "????.??.??")]
        return {"title": args.title or f"{h.get('White', '?')} vs. {h.get('Black', '?')}",
                "subtitle": args.subtitle or " · ".join(details),
                "moves": list(game.mainline_moves()), "result": h.get("Result", "*")}
    preset = GAMES[args.game]
    board, moves = chess.Board(), []
    for san in preset["moves"].split():
        moves.append(board.push_san(san))
    return {"title": args.title or preset["title"], "subtitle": args.subtitle or preset["subtitle"],
            "moves": moves, "result": board.result()}


def read_positions(model, moves):
    """Runs the network on every position before a move: full 4,096-way softmax plus legal-move shares."""
    board, positions = chess.Board(), []
    for move in moves:
        with torch.no_grad():
            logits = model(torch.from_numpy(encode_board(board)).unsqueeze(0))[0][0].numpy().astype(np.float64)
        policy = np.exp(logits - logits.max())
        policy /= policy.sum()
        legal = legal_move_indices(board)
        indices = list(legal)
        shares = policy[indices] / policy[indices].sum()
        share = dict(zip(indices, shares))
        top = indices[int(np.argmax(shares))]
        played = move.from_square * 64 + move.to_square
        positions.append({
            "policy": policy, "move": move, "san": board.san(move), "ply": board.ply(),
            "share": share.get(played, 0.0), "top_san": board.san(legal[top]), "top_share": share[top],
        })
        board.push(move)
    return positions


def colour_map(values, stops):
    stops = np.array(stops, float)
    x = np.clip(values, 0, 1) * (len(stops) - 1)
    i = np.minimum(x.astype(int), len(stops) - 2)
    t = (x - i)[..., None]
    return (stops[i] * (1 - t) + stops[i + 1] * t).round().astype(np.uint8)


def log_scale(p):
    return (np.log10(np.maximum(p, 1e-12)) - LOG_FLOOR) / -LOG_FLOOR


def square_origin(square, cell, gap):
    """Top-left pixel of a square's 8×8 block inside a tile (rank 8 at the top)."""
    return chess.square_file(square) * (8 * cell + gap), (7 - chess.square_rank(square)) * (8 * cell + gap)


def cell_origin(from_square, to_square, cell, gap):
    x, y = square_origin(from_square, cell, gap)
    return x + chess.square_file(to_square) * cell, y + (7 - chess.square_rank(to_square)) * cell


def render_tile(policy, cell, gap, theme):
    size = 64 * cell + 7 * gap
    tile = np.empty((size, size, 3), np.uint8)
    tile[:] = theme["bg"]
    colours = colour_map(log_scale(policy.reshape(64, 64)), theme["stops"])
    for from_square in range(64):
        # Inner board: row 0 is rank 8, column 0 is file a.
        inner = colours[from_square].reshape(8, 8, 3)[::-1]
        block = np.kron(inner, np.ones((cell, cell, 1), np.uint8))
        x, y = square_origin(from_square, cell, gap)
        tile[y:y + 8 * cell, x:x + 8 * cell] = block
    return Image.fromarray(tile)


def ring(draw, x, y, size, colour, bg, cell):
    gap, width = max(1, cell // 5), max(2, cell // 4)
    draw.rectangle([x - gap - width, y - gap - width, x + size + gap + width - 1, y + size + gap + width - 1],
                   fill=colour)
    draw.rectangle([x - gap, y - gap, x + size + gap - 1, y + size + gap - 1], fill=bg)


def wrap(text, face, width):
    lines, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if line and face.getlength(candidate) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + [line]


def paragraph(draw, xy, text, face, width, fill, leading=1.45):
    x, y = xy
    for line in wrap(text, face, width):
        draw.text((x, y), line, font=face, fill=fill)
        y += face.size * leading
    return y


def choose_grid(n, width, height, caption, gutter):
    """Columns and tile size that make the tiles as large as possible inside the area."""
    best = None
    for cols in range(2, 13):
        rows = math.ceil(n / cols)
        tile = min(width / (cols + gutter * (cols - 1)),
                   height / (rows * (1 + caption) + gutter * (rows - 1)))
        if best is None or tile > best[1]:
            best = (cols, tile)
    return best


def draw_key(draw, x, y, size, theme, faces):
    """A small diagram explaining the board-of-boards layout, using 1.Nf3 (g1 to f3)."""
    big = size / 8
    for i in range(9):
        draw.line([(x + i * big, y), (x + i * big, y + size)], fill=theme["muted"], width=1)
        draw.line([(x, y + i * big), (x + size, y + i * big)], fill=theme["muted"], width=1)
    from_square, to_square = chess.G1, chess.F3
    bx = x + chess.square_file(from_square) * big
    by = y + (7 - chess.square_rank(from_square)) * big
    draw.rectangle([bx, by, bx + big, by + big], fill=theme["stops"][2])
    small = big / 8
    cx = bx + chess.square_file(to_square) * small
    cy = by + (7 - chess.square_rank(to_square)) * small
    draw.rectangle([cx, cy, cx + small, cy + small], fill=theme["stops"][-1])
    draw.rectangle([cx - small, cy - small, cx + 2 * small, cy + 2 * small], outline=theme["accent"],
                   width=max(2, int(small / 3)))


def draw_colour_bar(img, draw, x, y, width, height, theme, faces):
    ramp = colour_map(np.linspace(0, 1, int(width)), theme["stops"])
    img.paste(Image.fromarray(np.repeat(ramp[None], int(height), axis=0)), (int(x), int(y)))
    for exponent, label in [(-6, "0.0001%"), (-4, "0.01%"), (-2, "1%"), (0, "100%")]:
        tx = x + (exponent - LOG_FLOOR) / -LOG_FLOOR * (width - 1)
        draw.line([(tx, y + height), (tx, y + height * 1.35)], fill=theme["muted"], width=max(1, int(height / 12)))
        label_width = faces["small"].getlength(label)
        lx = min(max(tx - label_width / 2, x), x + width - label_width)
        draw.text((lx, y + height * 1.5), label, font=faces["small"], fill=theme["muted"])


def render(game, positions, theme, paper, dpi, parameters, model_name):
    w_mm, h_mm = PAPERS_MM[paper]
    W, H = round(w_mm / 25.4 * dpi), round(h_mm / 25.4 * dpi)
    img = Image.new("RGB", (W, H), theme["bg"])
    draw = ImageDraw.Draw(img)
    margin = 0.075 * W
    inner = W - 2 * margin
    faces = {
        "label": font("mono", 0.0105 * W), "title": font("display", 0.062 * W),
        "subtitle": font("text", 0.0165 * W), "body": font("text", 0.0118 * W),
        "small": font("mono", 0.0095 * W), "result": font("text", 0.045 * W),
    }

    # Header.
    n = len(positions)
    y = margin
    draw.text((margin, y), "KNIGHTVISION  ·  POLICY QUILT", font=faces["label"], fill=theme["accent"])
    count = f"{n} positions  ·  {n * 4096:,} probabilities"
    draw.text((W - margin - faces["label"].getlength(count), y), count, font=faces["label"], fill=theme["muted"])
    y += faces["label"].size * 2.2
    draw.text((margin - 0.004 * W, y), game["title"], font=faces["title"], fill=theme["fg"])
    y += faces["title"].size * 1.18
    y = paragraph(draw, (margin, y), game["subtitle"], faces["subtitle"], inner, theme["muted"], 1.35)
    grid_top = y + 0.02 * H

    # Colophon at the bottom.
    colophon_h = 0.105 * H
    colophon_top = H - margin - colophon_h
    draw.line([(margin, colophon_top), (W - margin, colophon_top)], fill=theme["muted"], width=max(1, dpi // 150))

    # Tile grid.
    caption, gutter = 0.18, 0.09
    cols, target = choose_grid(n + 1, inner, colophon_top - 0.03 * H - grid_top, caption, gutter)
    # Cells are whole pixels so the print stays sharp; the gaps between inner boards absorb the rest.
    cell = max(1, int(target / (64 + 7 * 0.25)))
    gap = min(max(1, int((target - 64 * cell) / 7)), max(1, round(cell * 0.5)))
    tile = 64 * cell + 7 * gap
    step_x = (inner - tile) / (cols - 1) if cols > 1 else 0
    rows = math.ceil((n + 1) / cols)
    pitch_y = tile * (1 + caption + gutter)
    used_h = rows * tile * (1 + caption) + (rows - 1) * tile * gutter
    top = grid_top + (colophon_top - 0.03 * H - grid_top - used_h) / 2
    caption_face, note_face = font("mono", tile * 0.07), font("mono", tile * 0.056)

    for k, pos in enumerate(positions):
        x, y = round(margin + (k % cols) * step_x), round(top + (k // cols) * pitch_y)
        img.paste(render_tile(pos["policy"], cell, gap, theme), (x, y))
        move = pos["move"]
        cx, cy = cell_origin(move.from_square, move.to_square, cell, gap)
        ring(draw, x + cx, y + cy, cell, theme["stops"][-1], theme["bg"], cell)
        # Repaint the played cell, which the ring's inner fill covered.
        value = colour_map(log_scale(pos["policy"][move.from_square * 64 + move.to_square]), theme["stops"])
        draw.rectangle([x + cx, y + cy, x + cx + cell - 1, y + cy + cell - 1], fill=tuple(int(v) for v in value))

        number = pos["ply"] // 2 + 1
        label = f"{number}. {pos['san']}" if pos["ply"] % 2 == 0 else f"{number}… {pos['san']}"
        ty = y + tile + tile * 0.04
        draw.text((x, ty), label, font=caption_face, fill=theme["fg"])
        surprised = pos["san"] != pos["top_san"]
        share = "<0.1%" if pos["share"] < 0.0005 else "99%+" if pos["share"] >= 0.995 else f"{pos['share'] * 100:.1f}%"
        draw.text((x + tile - caption_face.getlength(share), ty), share, font=caption_face,
                  fill=theme["accent"] if surprised else theme["fg"])
        note = f"first choice {pos['top_san']} {pos['top_share'] * 100:.0f}%" if surprised else "its first choice"
        draw.text((x, ty + caption_face.size * 1.3), note, font=note_face, fill=theme["muted"])

    # The result, in the next free slot.
    k = n
    x, y = margin + (k % cols) * step_x, top + (k // cols) * pitch_y
    result = game["result"].replace("1/2-1/2", "½–½").replace("-", "–")
    rw = faces["result"].getlength(result)
    draw.text((x + (tile - rw) / 2, y + tile * 0.32), result, font=faces["result"], fill=theme["fg"])
    final = positions[-1]["san"]
    fw = caption_face.getlength(final)
    draw.text((x + (tile - fw) / 2, y + tile * 0.32 + faces["result"].size * 1.35), final,
              font=caption_face, fill=theme["muted"])

    # Colophon: how to read it, the colour scale, the data.
    cy = colophon_top + 0.016 * H
    col_w = (inner - 2 * 0.04 * W) / 3
    xs = [margin + i * (col_w + 0.04 * W) for i in range(3)]

    draw.text((xs[0], cy), "HOW TO READ A TILE", font=faces["small"], fill=theme["accent"])
    key = colophon_h * 0.52
    draw_key(draw, xs[0], cy + faces["small"].size * 2, key, theme, faces)
    paragraph(draw, (xs[0] + key + 0.012 * W, cy + faces["small"].size * 2),
              "Each tile is one position. Large squares: where a piece starts. The small board inside each: "
              "where the network would send it. Ringed: the move played. Key: 1.Nf3, g1 to f3.",
              faces["body"], col_w - key - 0.012 * W, theme["fg"])

    draw.text((xs[1], cy), "PROBABILITY, LOG SCALE", font=faces["small"], fill=theme["accent"])
    bar_y = cy + faces["small"].size * 2.2
    draw_colour_bar(img, draw, xs[1], bar_y, col_w, colophon_h * 0.16, theme, faces)
    paragraph(draw, (xs[1], bar_y + colophon_h * 0.42),
              "Colour shows the network's raw belief over all 4,096 start-and-end square pairs, legal or not. "
              "Captions give the played move's share among legal moves.",
              faces["body"], col_w, theme["fg"])

    draw.text((xs[2], cy), "THE NETWORK", font=faces["small"], fill=theme["accent"])
    facts = [
        f"ChessNet, {parameters / 1e6:.1f}M parameters ({model_name})",
        "No search: one forward pass per position",
        "About 935 Elo against Stockfish (±99)",
        f"Rendered {date.today().isoformat()} · KnightVision v2",
    ]
    fy = cy + faces["small"].size * 2
    for fact in facts:
        fy = paragraph(draw, (xs[2], fy), fact, faces["body"], col_w, theme["fg"], 1.4)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--game", choices=sorted(GAMES), default="opera")
    source.add_argument("--pgn", help="PGN file; renders its first game unless --game-index is given")
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument("--title")
    parser.add_argument("--subtitle")
    parser.add_argument("--paper", choices=sorted(PAPERS_MM), default="A2")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bleed", type=float, default=3.0, help="bleed on each edge in mm, filled with the background")
    parser.add_argument("--theme", choices=sorted(THEMES), default="night")
    parser.add_argument("--out", default=os.path.join("runs", "quilt"))
    args = parser.parse_args()

    state = torch.load(args.model_path, map_location="cpu")
    model = ChessNet()
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    parameters = sum(p.numel() for p in model.parameters())

    game = load_game(args)
    positions = read_positions(model, game["moves"])
    theme = THEMES[args.theme]
    poster = render(game, positions, theme, args.paper, args.dpi, parameters, os.path.basename(args.model_path))

    bleed = round(args.bleed / 25.4 * args.dpi)
    if bleed:
        page = Image.new("RGB", (poster.width + 2 * bleed, poster.height + 2 * bleed), theme["bg"])
        page.paste(poster, (bleed, bleed))
        poster = page

    os.makedirs(args.out, exist_ok=True)
    slug = args.game if not args.pgn else os.path.splitext(os.path.basename(args.pgn))[0]
    stem = os.path.join(args.out, f"{slug}-{args.paper}-{args.theme}")
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    poster.save(stem + ".png", dpi=(args.dpi, args.dpi), icc_profile=srgb)
    poster.save(stem + ".pdf", resolution=args.dpi, quality=100, subsampling=0)
    preview = poster.copy()
    preview.thumbnail((1600, 1600 * poster.height // poster.width), Image.LANCZOS)
    preview.save(stem + "-preview.jpg", quality=90)
    print(f"{stem}.png  {poster.width}×{poster.height}px, {args.paper} at {args.dpi} dpi with {args.bleed} mm bleed")


if __name__ == "__main__":
    main()

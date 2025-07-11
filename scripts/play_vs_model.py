# play_vs_model.py

import pygame as p
import torch
import os
try:
    import google.colab
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    BASE_DIR = "/content/drive/MyDrive/KnightVision"
except (ModuleNotFoundError, AttributeError):
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
from core.chessEngine import GameState
from core import chessMain
from ai import encode_board, encode_move
from ai.model import ChessNet

WIDTH = HEIGHT = 512  # chess board dimensions
DIMENSION = 8  # 8x8 board
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15  # for animations
IMAGES = {}

AI_MOVE_DELAY_MS = 400  # 4-second delay to allow the player to follow the game

def loadImages():
    pieces = ["wp", "wR", "wN", "wB", "wQ", "wK", "bp", "bR", "bN", "bB", "bQ", "bK"]
    for piece in pieces:
        IMAGES[piece.lower()] = p.transform.scale(p.image.load(os.path.join(BASE_DIR, "images", piece + ".png")), (SQ_SIZE, SQ_SIZE))

loadImages()
chessMain.IMAGES = IMAGES

def get_ai_move(gs, model):
    valid_moves = gs.getValidMoves()
    board_tensor = torch.tensor([encode_board(gs.board)]).float()
    with torch.no_grad():
        policy_logits, _ = model(board_tensor)
    policy = torch.softmax(policy_logits.squeeze(), dim=0).cpu().numpy()

    legal_indices = [encode_move(m.startRow, m.startCol, m.endRow, m.endCol) for m in valid_moves]
    legal_probs = [policy[i] for i in legal_indices]

    if sum(legal_probs) == 0:
        print("[WARNING] Model returned zero probability for all legal moves. Picking the first move.")
        return valid_moves[0]
    else:
        normalized_probs = [w / sum(legal_probs) for w in legal_probs]
        return valid_moves[normalized_probs.index(max(normalized_probs))]


def main():
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = GameState()

    def get_latest_checkpoint(checkpoint_dir=os.path.join(BASE_DIR, "checkpoints")):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
        if not checkpoint_files:
            raise FileNotFoundError("No checkpoint files (*.pth) found in the 'checkpoints' directory.")
        latest_file = max(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(checkpoint_dir, x)))
        return os.path.join(checkpoint_dir, latest_file)
        
    model = ChessNet()
    try:
        model_path = os.path.join(BASE_DIR, "checkpoints", "best_model_20250629_220833.pth")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load model checkpoint: {e}")
        return
    model.eval()
    running = True
    sq_selected = ()
    player_clicks = []
    player_turn = True  # Human plays white

    while running:
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif e.type == p.MOUSEBUTTONDOWN and player_turn:
                loc = p.mouse.get_pos()
                col = loc[0] // SQ_SIZE
                row = loc[1] // SQ_SIZE
                if sq_selected == (row, col):
                    sq_selected = ()
                    player_clicks = []
                else:
                    sq_selected = (row, col)
                    player_clicks.append(sq_selected)
                if len(player_clicks) == 2:
                    move = None
                    for m in gs.getValidMoves():
                        if m.startRow == player_clicks[0][0] and m.startCol == player_clicks[0][1] and \
                           m.endRow == player_clicks[1][0] and m.endCol == player_clicks[1][1]:
                            move = m
                    if move:
                        gs.makeMove(move)
                        chessMain.drawBoard(screen)
                        chessMain.highlightSquares(screen, gs, gs.getValidMoves(), sq_selected)
                        chessMain.drawPieces(screen, gs.board)
                        p.display.flip()
                        p.time.wait(200)  # slight delay to separate human move from AI thinking
                        player_turn = False
                    sq_selected = ()
                    player_clicks = []

        if not player_turn:
            ai_move = get_ai_move(gs, model)
            p.time.wait(AI_MOVE_DELAY_MS)
            gs.makeMove(ai_move)
            player_turn = True

        chessMain.drawBoard(screen)
        chessMain.highlightSquares(screen, gs, gs.getValidMoves(), sq_selected)
        chessMain.drawPieces(screen, gs.board)
        clock.tick(MAX_FPS)
        p.display.flip()


if __name__ == "__main__":
    # Ensure chessMain has access to IMAGES loaded here
    chessMain.IMAGES = IMAGES
    main()
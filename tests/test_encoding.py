import unittest
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'core'))
from chessEngine import GameState

try:
    import chess
    import numpy as np
    from ai.ai import (
        PIECE_PLANES, encode_board, encode_move, encode_chess_move,
        decode_move_index, legal_move_indices,
    )
    HAVE_ML_DEPS = True
except ImportError:
    HAVE_ML_DEPS = False


def reference_training_encoding(fen):
    """
    Frozen copy of the encoder the existing checkpoints were trained with
    (ChessPGNDataset.fen_to_tensor in scripts/train.py before v2). Do not change it:
    if encode_board stops matching it, the checkpoints no longer see what they learned.
    """
    board = chess.Board(fen)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_to_plane = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }
    for square, piece in board.piece_map().items():
        tensor[piece_to_plane[piece.symbol()]][7 - (square // 8)][square % 8] = 1.0
    return tensor


def find_engine_move(gs, chess_move):
    start = (7 - chess.square_rank(chess_move.from_square), chess.square_file(chess_move.from_square))
    end = (7 - chess.square_rank(chess_move.to_square), chess.square_file(chess_move.to_square))
    for move in gs.getValidMoves():
        if (move.startRow, move.startCol) == start and (move.endRow, move.endCol) == end:
            return move
    return None


def engine_from_fen(fen):
    """Builds a GameState for a position without castling rights."""
    board = chess.Board(fen)
    gs = GameState()
    for row in range(8):
        for col in range(8):
            piece = board.piece_at(chess.square(col, 7 - row))
            if piece is None:
                gs.board[row][col] = "--"
            else:
                letter = 'p' if piece.piece_type == chess.PAWN else piece.symbol().upper()
                gs.board[row][col] = ('w' if piece.color == chess.WHITE else 'b') + letter
    gs.whiteToMove = board.turn == chess.WHITE
    gs.whiteKingLocation = (7 - chess.square_rank(board.king(chess.WHITE)), chess.square_file(board.king(chess.WHITE)))
    gs.blackKingLocation = (7 - chess.square_rank(board.king(chess.BLACK)), chess.square_file(board.king(chess.BLACK)))
    gs.wKingMoved = gs.bKingMoved = True
    if board.ep_square is not None:
        gs.enPassantPossible = (7 - chess.square_rank(board.ep_square), chess.square_file(board.ep_square))
    return gs


@unittest.skipUnless(HAVE_ML_DEPS, "needs numpy, python-chess and torch")
class TestEncoding(unittest.TestCase):
    def test_board_matches_training_encoder(self):
        fens = [
            chess.STARTING_FEN,
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            "8/2k5/3p4/p2P1p2/P2P1P2/8/5K2/8 b - - 0 1",
        ]
        for fen in fens:
            np.testing.assert_array_equal(encode_board(fen), reference_training_encoding(fen))
            np.testing.assert_array_equal(encode_board(chess.Board(fen)), reference_training_encoding(fen))

    def test_engine_board_matches_chess_board(self):
        np.testing.assert_array_equal(encode_board(GameState().board), encode_board(chess.Board()))

    def test_plane_layout(self):
        encoded = encode_board(chess.Board())
        self.assertEqual(encoded[PIECE_PLANES.index('K'), 7, 4], 1.0)  # white king on e1
        self.assertEqual(encoded[PIECE_PLANES.index('k'), 0, 4], 1.0)  # black king on e8
        self.assertEqual(encoded[PIECE_PLANES.index('P'), 6].sum(), 8.0)  # white pawns on rank 2
        self.assertEqual(encoded.sum(), 32.0)

    def test_known_move_index(self):
        e2e4 = chess.Move.from_uci("e2e4")
        self.assertEqual(encode_chess_move(e2e4), chess.E2 * 64 + chess.E4)
        self.assertEqual(encode_move(6, 4, 4, 4), encode_chess_move(e2e4))

    def test_decode_round_trip(self):
        for index in range(4096):
            self.assertEqual(encode_move(*decode_move_index(index)), index)

    def test_queen_promotion_preferred(self):
        board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
        legal = legal_move_indices(board)
        self.assertEqual(legal[chess.A7 * 64 + chess.A8].promotion, chess.QUEEN)

    def test_engine_and_python_chess_agree_in_tricky_positions(self):
        fens = {
            "knight check from two ranks up, one file right": "8/4n2p/1Q2pk2/P1P4P/3Rpr2/3Bn1R1/1N1P2P1/2BK4 w - - 4 52",
            "a pawn push is not an attack": "b3n3/r6p/2k4n/5p1P/r2Pp3/B4pP1/3P4/RBR4K b - - 1 47",
            "king cannot step into a pawn's capture square": "8/8/8/8/3p4/8/4K3/7k w - - 0 1",
            "white pawn pinned on its file can still push": "2K5/8/8/2P5/8/5R2/4R3/2r3k1 w - - 25 127",
            "black pawn pinned on its file can still push": "2r5/p7/8/pPp1R3/8/4p2P/8/1K2k3 b - - 5 73",
            "en passant captures the checking pawn": "1r2qbnr/p1pk4/n4p2/3Pp1pp/bpPK2P1/1P3N1P/PBB2P2/RN5R w - e6 0 23",
            "en passant that opens the rank onto the king": "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",
        }
        for name, fen in fens.items():
            gs = engine_from_fen(fen)
            engine_indices = {encode_move(m.startRow, m.startCol, m.endRow, m.endCol) for m in gs.getValidMoves()}
            self.assertEqual(engine_indices, set(legal_move_indices(chess.Board(fen))), name)

    def test_engine_and_python_chess_agree_through_games(self):
        """
        Plays random games in python-chess and mirrors them in GameState. At every ply both
        sides must produce the same board tensor and the same set of legal move indices.
        """
        rng = random.Random(7)
        for _ in range(20):
            board = chess.Board()
            gs = GameState()
            for _ in range(120):
                if board.is_game_over():
                    break
                np.testing.assert_array_equal(encode_board(gs.board), encode_board(board), board.fen())
                engine_indices = {encode_move(m.startRow, m.startCol, m.endRow, m.endCol) for m in gs.getValidMoves()}
                legal = legal_move_indices(board)
                self.assertEqual(engine_indices, set(legal), board.fen())

                chess_move = legal[rng.choice(sorted(legal))]
                engine_move = find_engine_move(gs, chess_move)
                gs.makeMove(engine_move)
                board.push(chess_move)


if __name__ == "__main__":
    unittest.main()

import unittest
from chessEngine import GameState, Move


def _base_state():
    """Return a GameState with only the kings on the board."""
    gs = GameState()
    gs.board = [["--" for _ in range(8)] for _ in range(8)]
    gs.board[7][4] = 'wK'
    gs.board[0][4] = 'bK'
    gs.whiteKingLocation = (7, 4)
    gs.blackKingLocation = (0, 4)
    return gs


class TestEnPassant(unittest.TestCase):
    def test_white_en_passant_capture_and_undo(self):
        gs = _base_state()
        gs.board[3][4] = 'wp'  # white pawn on e5
        gs.board[1][3] = 'bp'  # black pawn on d7
        gs.whiteToMove = False  # Black to move first

        # Black pawn moves two squares from d7 to d5
        move1 = Move((1, 3), (3, 3), gs.board)
        gs.makeMove(move1)
        self.assertEqual(gs.enPassantPossible, (2, 3))

        # White captures en passant with pawn from e5 to d6
        move2 = Move((3, 4), (2, 3), gs.board, isEnPassantMove=True)
        gs.makeMove(move2)
        self.assertEqual(gs.enPassantPossible, ())

        # Undo the en passant capture
        gs.undoMove()
        self.assertEqual(gs.enPassantPossible, (2, 3))
        self.assertEqual(gs.board[3][3], 'bp')
        self.assertEqual(gs.board[3][4], 'wp')

        # Undo the pawn double move
        gs.undoMove()
        self.assertEqual(gs.enPassantPossible, ())
        self.assertEqual(gs.board[1][3], 'bp')
        self.assertEqual(gs.board[3][4], 'wp')

    def test_black_en_passant_capture_and_undo(self):
        gs = _base_state()
        gs.board[6][3] = 'wp'  # white pawn on d2
        gs.board[4][4] = 'bp'  # black pawn on e4

        # White pawn moves two squares from d2 to d4
        move1 = Move((6, 3), (4, 3), gs.board)
        gs.makeMove(move1)
        self.assertEqual(gs.enPassantPossible, (5, 3))

        # Black captures en passant with pawn from e4 to d3
        move2 = Move((4, 4), (5, 3), gs.board, isEnPassantMove=True)
        gs.makeMove(move2)
        self.assertEqual(gs.enPassantPossible, ())

        # Undo the en passant capture
        gs.undoMove()
        self.assertEqual(gs.enPassantPossible, (5, 3))
        self.assertEqual(gs.board[4][4], 'bp')
        self.assertEqual(gs.board[4][3], 'wp')

        # Undo the pawn double move
        gs.undoMove()
        self.assertEqual(gs.enPassantPossible, ())
        self.assertEqual(gs.board[6][3], 'wp')
        self.assertEqual(gs.board[4][4], 'bp')


if __name__ == '__main__':
    unittest.main()

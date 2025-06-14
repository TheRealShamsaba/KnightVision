import unittest
from chessEngine import GameState, Move

class TestPromotion(unittest.TestCase):
    def test_white_pawn_promotion_to_queen(self):
        gs = GameState()
        gs.board[1][0] = "--"
        gs.board[6][0] = "--"
        gs.board[1][0] = "wp"
        gs.whiteToMove = True

        move = Move((1, 0), (0, 0), gs.board)
        move.isPawnPromotion = True  # Set manually
        move.promotionChoice = "Q"
        gs.makeMove(move)

        self.assertEqual(gs.board[0][0], "wQ")
        gs.undoMove()
        self.assertEqual(gs.board[1][0], "wp")

    def test_black_pawn_promotion_to_knight(self):
        gs = GameState()
        gs.board[6][7] = "--"
        gs.board[1][7] = "--"
        gs.board[6][7] = "bp"
        gs.whiteToMove = False

        move = Move((6, 7), (7, 7), gs.board)
        move.isPawnPromotion = True  # Set manually
        move.promotionChoice = "N"
        gs.makeMove(move)

        self.assertEqual(gs.board[7][7], "bN")
        gs.undoMove()
        self.assertEqual(gs.board[6][7], "bp")

if __name__ == "__main__":
    unittest.main()
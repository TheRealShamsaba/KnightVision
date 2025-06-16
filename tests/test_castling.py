import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from chessEngine import GameState, Move

class TestCastling(unittest.TestCase):
    def _setup_castling_position(self):
        gs = GameState()
        gs.board = [
            
            ["bR", "--", "--", "--", "bK", "--", "--", "bR"],
            ["--", "--", "--", "--", "--", "--", "--", "--",],
            ["--", "--", "--", "--", "--", "--", "--", "--",],
            ["--", "--", "--", "--", "--", "--", "--", "--",],
            ["--", "--", "--", "--", "--", "--", "--", "--",],
            ["--", "--", "--", "--", "--", "--", "--", "--",],
            ["--", "--", "--", "--", "--", "--", "--", "--",],
            ["wR", "--", "--", "--", "wK", "--", "--", "wR"]         
        ]
        gs.whiteKingLocation = (7, 4)
        gs.blackKingLocation = (0 ,4)
        gs.wKingMoved = False
        gs.bKingMoved = False
        gs.wRookKingsideMoved = False
        gs.wRookQueensideMoved = False
        gs.bRookKingsideMoved = False
        gs.bRookQueensideMoved = False
        return gs
    
    def test_white_kingside_castling(self):
        gs = self._setup_castling_position()
        validMoves = gs.getValidMoves()
        move =  Move((7,4), (7,6), gs.board, isCastleMove = True)
        self.assertIn(move, validMoves)
        
    def test_white_queenside_castling(self):
        gs = self._setup_castling_position()
        validMoves = gs.getValidMoves()
        move = Move((7, 4), (7, 2), gs.board, isCastleMove=True)
        self.assertIn(move, validMoves)

    def test_black_kingside_castling(self):
        gs = self._setup_castling_position()
        gs.whiteToMove = False
        validMoves = gs.getValidMoves()
        move = Move((0, 4), (0, 6), gs.board, isCastleMove=True)
        self.assertIn(move, validMoves)

    def test_black_queenside_castling(self):
        gs = self._setup_castling_position()
        gs.whiteToMove = False
        validMoves = gs.getValidMoves()
        move = Move((0, 4), (0, 2), gs.board, isCastleMove=True)
        self.assertIn(move, validMoves)
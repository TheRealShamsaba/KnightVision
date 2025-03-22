"""
This class is responsible for starting all the information about the cuttent state of a chess game. 
it will also be responsible for determining the valid moves at thte cureent state and it will also keep a move log
"""

class gameState():
    def _init_(self):
        #The board is 8*8 2d list, each element of the list has a 2 characters
        #the firsr char representes the color of the piece, 'b' , 'w'
        #the second char represents the type of the piece
        # and the "--" is the emty spaces
        self.board = [
            ["bR" , "bN" , "bB" , "bQ" , "bK" , "bB" , "bN" , "bR"],
            ["bp" , "bp" , "bp" , "bp" , "bp" , "bp" , "bp" , "bp"],
            ["--" , "--" , "--" , "--" , "--" , "--" , "--" , "--"],
            ["--" , "--" , "--" , "--" , "--" , "--" , "--" , "--"],
            ["--" , "--" , "--" , "--" , "--" , "--" , "--" , "--"],
            ["--" , "--" , "--" , "--" , "--" , "--" , "--" , "--"],
            ["wp" , "wp" , "wp" , "wp" , "wp" , "wp" , "wp" , "wp"],
            ["wR" , "wN" , "wB" , "wQ" , "wK" , "wB" , "wN" , "wR"],
        ]
        self.whiteToMove = True
        self.moveLog = []
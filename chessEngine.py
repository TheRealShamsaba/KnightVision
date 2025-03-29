"""
This class is responsible for starting all the information about the cuttent state of a chess game. 
it will also be responsible for determining the valid moves at thte cureent state and it will also keep a move log
"""

class GameState():
    def __init__(self):
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
# takes a move as a paramiter and sexcutes it (this doesnt work for casteling, pawn promotion and etc)

    def makeMove(self, move):
        self.board[move.startRow][move.startCol] = "--"
        self.board[move.endRow][move.endCol] =  move.pieceMoved
        self.moveLog.append(move) # log the game
        self.whiteToMove = not self.whiteToMove #swap players
        
    """
    undo the last move made
    """
    def undoMove(self):
        if len(self.moveLog) != 0: # make sure that there is a move to undo
            move = self.moveLog.pop()
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove # switch turns back
            
    '''
    all moves considering checks
    '''
    def getVaildMoves(self):
        return self.getAllPossibleMoves() # for now will not worry for checks
        
    '''
    all moves without considering checks
    '''
    
    def getAllPossibleMoves(self):
        move = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn: self.board[r][c][0]
                if (turn == 'w' and self.whiteToMove) and (turn =='b' and not self.whiteToMove):
                    piece = self.board[r][c][1]
                    
                
    
        
        
class Move():
    # maps keys in values
    # key : value
    ranksToRows = {"1" : 7, "2": 6, "3": 5, "4" : 4,
                   "5" : 3, "6" : 2, "7" : 1, "8" : 0}
    rowsToRanks = {v : k for k, v in ranksToRows.items()}
    filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3,
                   "e": 4, "f": 5, "g":6, "h": 7}
    colsToFiles = {v: k for k, v in filesToCols.items()}
    
    def __init__(self, startSq, endSq, board):
         self.startRow = startSq[0]
         self.startCol = startSq[1]
         self.endRow = endSq[0]
         self.endCol = endSq[1]
         self.pieceMoved = board[self.startRow][self.startCol]
         self.pieceCaptured = board[self.endRow][self.endCol]
         
    def getChessNotaion(self):
        # you can add to make this like real chess notation
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(self.endRow, self.endCol)
        
    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]
        
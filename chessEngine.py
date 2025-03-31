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
            ["bp" , "bp" , "bp" , "--" , "bp" , "bp" , "bp" , "bp"],
            ["--" , "--" , "--" , "--" , "--" , "--" , "--" , "--"],
            ["--" , "--" , "--" , "--" , "--" , "--" , "--" , "--"],
            ["--" , "--" , "--" , "--" , "--" , "--" , "--" , "--"],
            ["--" , "--" , "--" , "bp" , "--" , "--" , "--" , "--"],
            ["wp" , "wp" , "wp" , "wp" , "wp" , "wp" , "wp" , "wp"],
            ["wR" , "wN" , "wB" , "wQ" , "wK" , "wB" , "wN" , "wR"],]
        self.moveFunctions = {'p': self.getPawnMoves, 'R': self.getRookMoves, 'N': self.getKnightMoves,
                              'B': self.getBishopMoves, 'Q': self.getQueenMoves, 'K': self.getKingMoves}
        
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
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]
                if (turn == 'w' and self.whiteToMove) or (turn =='b' and not self.whiteToMove):
                    piece = self.board[r][c][1]
                    self.moveFunctions[piece](r, c, moves) # calls the appopriate move function based on piece types
        return moves 
                        
    '''
    get all the pawns moves for the pawn loacted at row, col and these moves to the lost
    '''
    def getPawnMoves(self, r, c, moves):
        if self.whiteToMove: # white pawn moves
            if self.board[r-1][c] == "--" : # 1 squre pawn advance
                moves.append(Move((r, c), (r-1, c), self.board))
                if r == 6  and self.board[r-2][c] == "--": # 2 square pawn advance
                     moves.append(Move((r, c), (r-2, c), self.board))
            if c-1 >= 0: # captures to the left 
                if self.board[r-1][c-1][0] == 'b' :# enemy piece to capture
                    moves.append(Move((r, c), (r-1, c-1), self.board))
            if c+1 < 7 : # captures to the right
                if self.board[r-1][c+1][0] == 'b':
                    moves.append(Move((r, c), (r-1, c+1), self.board))
        else: # black pawn moves
            pass
    
    '''
    get all the rook moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getRookMoves(self, r, c, moves):
        pass
                
    '''
    get all the Knight moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getKnightMoves(self, r, c, moves):
        pass
                
    '''
    get all the Bishop moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getBishopMoves(self, r, c, moves):
        pass
    
    '''
    get all the Queen moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getQueenMoves(self, r, c, moves):
        pass
    
    '''
    get all the King moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getKingMoves(self, r, c, moves):
        pass
                
                
                
        
        
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
         self.moveID = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
         print(self.moveID)
         
         
    '''
    overriding the equals method
    '''
    
    def __eq__(self, other):
        if isinstance(other,Move):
            return self.moveID == other.moveID
        return False
            
         
         
    def getChessNotaion(self):
        # you can add to make this like real chess notation
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(self.endRow, self.endCol)
        
    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]
        
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
            ["wR" , "wN" , "wB" , "wQ" , "wK" , "wB" , "wN" , "wR"],]
        self.moveFunctions = {'p': self.getPawnMoves, 'R': self.getRookMoves, 'N': self.getKnightMoves,
                              'B': self.getBishopMoves, 'Q': self.getQueenMoves, 'K': self.getKingMoves}
        
        self.whiteToMove = True
        self.moveLog = []
        self.whiteKingLocation = (7,4)
        self.blackKingLocation = (0,4)
        self.insideSqureUnderAttack = False

            
# takes a move as a paramiter and sexcutes it (this doesnt work for casteling, pawn promotion and etc)

    def makeMove(self, move):
        self.board[move.startRow][move.startCol] = "--"
        self.board[move.endRow][move.endCol] =  move.pieceMoved
        self.moveLog.append(move) # log the game
        self.whiteToMove = not self.whiteToMove #swap players
        # upadte the kinds location
        if move.pieceMoved == 'wK':
            self.whiteKingLocation = (move.endRow , move.endCol)
        elif move.pieceMoved == 'bK':
            self.blackKingLocation = (move.endRow , move.endCol)
        if move.isPawnPromotion:
            self.board[move.endRow][move.endCol] = move.pieceMoved[0] + move.promotionChoice
        
    """
    undo the last move made
    """
    def undoMove(self):
        if len(self.moveLog) != 0: # make sure that there is a move to undo
            move = self.moveLog.pop()
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            self.whiteToMove = not self.whiteToMove # switch turns back
            # upadte the kings location
            if move.pieceMoved == 'wK':
                self.whiteKingLocation = (move.startRow , move.startCol)
            elif move.pieceMoved == 'bK':
                self.blackKingLocation = (move.startRow , move.startCol)
            if move.isPawnPromotion:
                self.board[move.startRow][move.startCol] = move.pieceMoved
                self.board[move.endRow][move.endCol] = move.pieceCaptured

    '''
    all moves considering checks
    '''
    def getValidMoves(self):
        moves = []
        inCheck, pins, checks = self.checkForPinsAndChecks()

        if self.whiteToMove:
            kingRow, kingCol = self.whiteKingLocation
        else:
            kingRow, kingCol = self.blackKingLocation

        if inCheck:
            if len(checks) == 1:
                moves = self.getAllPossibleMoves(pins)
                check = checks[0]
                checkRow, checkCol, checkDirRow, checkDirCol = check
                pieceChecking = self.board[checkRow][checkCol]
                validSquares = []

                if pieceChecking[1] == "N":
                    validSquares = [(checkRow, checkCol)]
                else:
                    for i in range(1, 8):
                     if inCheck:
                        square = (kingRow + checkDirRow * i, kingCol + checkDirCol * i)
                        print("check detected?", inCheck, "checks:", checks)
                        validSquares.append(square)
                        if square == (checkRow, checkCol):
                            break

                newMoves = []
                for move in moves:
                    if move.pieceMoved[1] == 'K':
                        if not self.squareUnderAttack(move.endRow, move.endCol):
                            newMoves.append(move)
                    elif (move.endRow, move.endCol) in validSquares:
                        newMoves.append(move)
                moves = newMoves
            else:
                moves = []
                self.getKingMoves(kingRow, kingCol, moves)
        else:
            moves = self.getAllPossibleMoves(pins)

        return moves
                    
                
        
    
    
    def checkForPinsAndChecks (self):
        pins = []
        checks = []
        inCheck = False
        
        if self.whiteToMove:
            enemyColor = 'b'
            allyColor = 'w'
            kingRow , kingCol = self.whiteKingLocation
        else:
            enemyColor = 'w'
            allyColor = 'b'
            kingRow, kingCol = self.blackKingLocation
        # checks directions for pins / checks from sliding pieces
        directions = [(-1,0), (0,-1), (1,0), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        for d in directions:
            possiblePin = ()
            for i in range(1, 8):
                endRow = kingRow + d[0] * i
                endCol = kingCol + d[1] * i
                if not (0 <= endRow < 8 and 0 <= endCol < 8):
                    break
                endPiece = self.board[endRow][endCol]
                if endPiece != "--":
                    if endPiece[0] == allyColor:
                        if possiblePin == ():
                            possiblePin = (endRow, endCol, d[0], d[1])
                        else:
                            break
                    elif endPiece[0] == enemyColor:
                        pieceType = endPiece[1]
                        print(f"Checking piece {pieceType} at {(endRow, endCol)} in direction {d}")
                        if ((d in [(-1, 0), (1, 0), (0, -1), (0, 1)] and (pieceType == 'R' or pieceType == 'Q')) or
                            (d in [(-1, -1), (-1, 1), (1, -1), (1, 1)] and (pieceType == 'B' or pieceType == 'Q')) or
                            (i == 1 and pieceType == 'p' and
                             ((enemyColor == 'w' and d in [(1, -1), (1, 1)]) or
                              (enemyColor == 'b' and d in [(-1, -1), (-1, 1)])))):
                            if possiblePin == (): 
                                inCheck = True
                                checks.append((endRow, endCol, d[0], d[1]))
                            else:
                                pins.append(possiblePin)
                            break
                        else:
                            break
                else:
                    continue
            # checks for knight checks
        knightMoves = [(-2,-1), (-1, -2), (-1, 2), 
                           (1, -2), (2,-1), (1,2), (2,1)]
        for m in knightMoves:
            endRow = kingRow + m[0]
            endCol= kingCol + m[1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] == enemyColor and endPiece[1] == "N":
                    inCheck = True
                    checks.append((endRow, endCol, m[0], m[1]))
        return inCheck, pins, checks
            
                       
                

    def inCheck(self):   
        if self.whiteToMove:
             print('Checking if white is in check')
             return self.squareUnderAttack(self.whiteKingLocation[0], self.whiteKingLocation[1])
        else:
             print('Checking if black is in check')
             return self.squareUnderAttack(self.blackKingLocation[0], self.blackKingLocation[1])
        
        
    '''
    determine if the enemy can attack the square r ,c 
    '''    
    def squareUnderAttack(self, r, c):
        if self.insideSqureUnderAttack:
            return False
        self.insideSqureUnderAttack = True

        originalTurn = self.whiteToMove
        self.whiteToMove = not originalTurn
        oppMoves = self.getAllPossibleMoves()

        self.whiteToMove = originalTurn
        self.insideSqureUnderAttack = False

        for move in oppMoves:
            if move.endRow == r and move.endCol == c:
                return True
        return False

    def getAllPossibleMovesRaw(self, ignoreKing=False):
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
             turn = self.board[r][c][0]
             if (turn == 'w' and self.whiteToMove) or (turn == 'b' and not self.whiteToMove):
                piece = self.board[r][c][1]
                if ignoreKing and piece == 'K':
                    continue
                self.moveFunctions[piece](r, c, moves)
        return moves
    
    '''
    all moves without considering checks
    '''
    
    def getAllPossibleMoves(self, pins=[]):
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]
                if (turn == 'w' and self.whiteToMove) or (turn =='b' and not self.whiteToMove):
                    piece = self.board[r][c][1]
                    self.addPieceMovesConsideringPins(piece, r, c, moves, pins)  # Updated line
        return moves 
                        
    '''
    get all the pawns moves for the pawn located at row, col and add these moves to the list
    Now considers pinDirection if specified.
    '''
    def getPawnMoves(self, r, c, moves, pinDirection=None):
        if self.whiteToMove:
            moveAmount = -1
            startRow = 6
            enemyColor = 'b'
        else:
            moveAmount = 1
            startRow = 1
            enemyColor = 'w'

        # Move forward (only if not pinned or pinned vertically)
        if pinDirection is None or pinDirection == (moveAmount, 0):
            if 0 <= r + moveAmount < 8 and self.board[r + moveAmount][c] == "--":
                moves.append(Move((r, c), (r + moveAmount, c), self.board))
                if r == startRow and self.board[r + 2 * moveAmount][c] == "--":
                    moves.append(Move((r, c), (r + 2 * moveAmount, c), self.board))

        # Captures (only if not pinned or pinned diagonally)
        for dc in [-1, 1]:
            if 0 <= c + dc < 8:
                if pinDirection is None or pinDirection == (moveAmount, dc):
                    if 0 <= r + moveAmount < 8 and self.board[r + moveAmount][c + dc][0] == enemyColor:
                        moves.append(Move((r, c), (r + moveAmount, c + dc), self.board))
                    
    '''
    get all the rook moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getRookMoves(self, r, c, moves):
        rookMoves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        enemyColor = 'b' if self.whiteToMove else 'w'
        for k in rookMoves:
            for i in range(1,8):
                endRow = r + k[0]*i
                endCol = c + k[1]*i
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPeace = self.board[endRow][endCol]
                    if endPeace == "--":
                        moves.append(Move((r,c), (endRow,endCol), self.board))
                    elif endPeace[0] ==enemyColor:
                        moves.append(Move((r,c), (endRow, endCol), self.board))
                        break
                    else:
                        break
                else:
                    break
            
                
    '''
    get all the Knight moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getKnightMoves(self, r, c, moves):
        knightMoves = [(-2, -1),(-1, -2),(-2, 1), (-1, 2), 
                       (1, -2), (2, -1), (1,2), (2,1) ]
        allyColor = 'w' if self.whiteToMove else 'b'
        for m in knightMoves:
            endRow = r + m[0]
            endCol = c + m[1]
           
            if 0 <= endRow < 8 and 0 <= endCol < 8:  # stay on board
                endPiece = self.board[endRow][endCol]
                # move if square is empty or has an enemy
                if endPiece == "--" or endPiece[0] != allyColor:
                    moves.append(Move((r, c), (endRow, endCol), self.board))   
    '''
    get all the Bishop moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getBishopMoves(self, r, c, moves):
        bishopMoves = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        enemyColor = 'b' if self.whiteToMove else 'w'
        for d in bishopMoves:
            for i in range(1, 8):
                endRow = r+ d[0] * i
                endCol = c + d[1] * i 
                if 0 <= endRow < 8 and 0 <= endCol < 8 : 
                    endPiece = self.board[endRow][endCol]
                    if endPiece == "--":
                        moves.append(Move((r,c),(endRow, endCol), self.board))
                    elif endPiece[0] == enemyColor:
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                        break
                    else:
                        break
                        
    '''
    get all the Queen moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getQueenMoves(self, r, c, moves):
        self.getRookMoves(r, c ,moves)
        self.getBishopMoves(r , c, moves)
    
    '''
    get all the King moves for the rook lacted at rowm col and these moves in the list
    '''   
    def getKingMoves(self, r, c, moves):
        kingMoves = [(-1, -1), (-1, 0), (-1, 1),
              (0, -1),          (0, 1),
              (1, -1),  (1, 0), (1, 1)]
        allyColor = 'w' if self.whiteToMove else 'b'
        for d in kingMoves:
            endRow = r + d[0]
            endCol = c + d[1]
            if 0 <= endRow < 8 and 0 <= endCol < 8:
                endPiece = self.board[endRow][endCol]
                if endPiece == "--" or endPiece[0] != allyColor:
                    # Only add the king move if the destination square is not under attack
                    originalSqure = self.board[endRow][endCol]
                    self.board[r][c] = "--"
                    self.board[endRow][endCol] = ('wK' if self.whiteToMove else 'bK')
                    originalKingLoc = self.whiteKingLocation if self.whiteToMove else self.blackKingLocation
                    if self.whiteToMove:
                        self.whiteKingLocation = (endRow, endCol)
                    else:
                        self.blackKingLocation = (endRow, endCol)
                    inCheck = self.squareUnderAttack(endRow, endCol)
                    self.board[r][c] = self.board[endRow][endCol]
                    self.board[endRow][endCol] = originalSqure
                    if self.whiteToMove:
                        self.whiteKingLocation = originalKingLoc
                    else:
                        self.blackKingLocation = originalKingLoc
                    if not inCheck:
                        moves.append(Move((r,c), (endRow, endCol), self.board))
                    
                    
    def addPieceMovesConsideringPins(self, piece, r, c, moves, pins):
        isPinned = False
        pinDirection = ()
        for i in range(len(pins)-1, -1, -1):
            if pins[i][0] == r and pins[i][1] == c:
                isPinned = True
                pinDirection = (pins[i][2], pins[i][3])
                break

        if isPinned:
            if piece == "N":
                return  # Knight is pinned; can't move
            else:
                tempMoves = []
                if piece == "p":
                    self.getPawnMoves(r, c, tempMoves, pinDirection)
                else:
                    self.moveFunctions[piece](r, c, tempMoves)
                for m in tempMoves:
                    moveDirection = (m.endRow - r, m.endCol - c)
                    if moveDirection[0] * pinDirection[1] == moveDirection[1] * pinDirection[0]:
                        moves.append(m)
        else:
            if piece == "p":
                self.getPawnMoves(r, c, moves)
            else:
                self.moveFunctions[piece](r, c, moves)

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
         self.isPawnPromotion = False
         if self.pieceMoved[1] == 'p':
             if (self.pieceMoved[0] == 'w' and self.endRow == 0) or \
                (self.pieceMoved[0] == 'b' and self.endRow == 7):
                 self.isPawnPromotion = True
         self.moveID = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
         self.promotionChoice = 'Q'  # default promotion
         
    '''
    overriding the equals method
    '''
    
    def __eq__(self, other):
        if isinstance(other,Move):
            return self.moveID == other.moveID
        return False
            
         
         
    def getChessNotation(self):
        # you can add to make this like real chess notation
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(self.endRow, self.endCol)
        
    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]


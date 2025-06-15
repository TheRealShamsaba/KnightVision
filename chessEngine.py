
"""
 + this module contains the :class:`GameState` class that hold information
 + about the current state of the chess game. it also provides methods to compute
 + valid moves and keep a move log
"""

import logging

logger = logging.getLogger(__name__)

# CastleRights class definition
class CastleRights:
    def __init__(self, wks, wqs, bks, bqs):
        self.wks = wks
        self.wqs = wqs
        self.bks = bks
        self.bqs = bqs

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
        self.moveFunctions = {
            'p': self.getPawnMoves, 'P': self.getPawnMoves,
            'r': self.getRookMoves, 'R': self.getRookMoves,
            'n': self.getKnightMoves, 'N': self.getKnightMoves,
            'b': self.getBishopMoves, 'B': self.getBishopMoves,
            'q': self.getQueenMoves, 'Q': self.getQueenMoves,
            'k': self.getKingMoves,  'K': self.getKingMoves
}
        
        self.whiteToMove = True
        self.moveLog = []
        self.whiteKingLocation = (7,4)
        self.blackKingLocation = (0,4)
        self.insideSquareUnderAttack = False
        # Track checkmate and stalemate
        self.checkMate = False
        self.staleMate = False
        # Castling rights tracking
        self.wKingMoved = False
        self.bKingMoved = False
        self.wRookKingsideMoved = False
        self.wRookQueensideMoved = False
        self.bRookKingsideMoved = False
        self.bRookQueensideMoved = False
        self.enPassantPossible = ()  # coordinates where en passant is possible
        self.enPassantPossibleLog = []

        # Draw detection support
        self.moveLogHistory = []
        self.boardHistory = {}
        # For 50-move rule
        self.halfMoveClock = 0
        # For threefold repetition
        self.boardStateCounter = {}
        self.draw50 = False
        self.drawRepetition = False
    def loadFEN(self, fen):
        parts = fen.split()
        board_part = parts[0]
        turn_part = parts[1]
        castling_part = parts[2]
        en_passant_part = parts [3]
        
        rows = board_part.split('/')
        for r in range(8):
            row = []
            for char in rows[r]:
                if char.isdigit():
                    row.extend(['--'] * int(char))
                else:
                    color = 'w' if char.isupper() else 'b'
                    piece = char.upper()
                    row.append(color + piece)
            self.board[r] = row
        
        self.whiteToMove = (turn_part == "w")
        
        
        if not hasattr(self, 'castleRights'):
            self.castleRights = CastleRights(False, False, False, False)
        self.castleRights.wks = 'K' in castling_part
        self.castleRights.bks = 'k' in castling_part
        self.castleRights.wqs = 'Q' in castling_part
        self.castleRights.bqs = 'q' in castling_part
        
        if en_passant_part != '-':
            file = ord(en_passant_part[0]) - ord('a')
            rank = 8 - int(en_passant_part[1])
            self.enPassantPossible = (rank, file)
        else:
            self.enPassantPossible = ()
        # 5. Clear move history
        self.moveLog = []
        self.enPassantPossibleLog = []

            
    # takes a move as a parameter and executes it (this doesn't work for castling, pawn promotion and etc)

    def makeMove(self, move):
        # save en passant state for undo functiality
        self.enPassantPossibleLog.append(self.enPassantPossible)
        # Store previous halfMoveClock for undo
        if not hasattr(self, 'halfMoveClockLog'):
            self.halfMoveClockLog = []
        self.halfMoveClockLog.append(self.halfMoveClock)
        self.board[move.startRow][move.startCol] = "--"
        self.board[move.endRow][move.endCol] = move.pieceMoved
        if move.pieceMoved == 'wK':
            self.wKingMoved = True
        elif move.pieceMoved == 'bK':
            self.bKingMoved = True
        elif move.pieceMoved == 'wR':
            if move.startRow == 7 and move.startCol == 0:
                self.wRookQueensideMoved = True
            elif move.startRow == 7 and move.startCol == 7:
                self.wRookKingsideMoved = True
        elif move.pieceMoved == 'bR':
            if move.startRow == 0 and move.startCol == 0:
                self.bRookQueensideMoved = True
            elif move.startRow == 0 and move.startCol == 7:
                self.bRookKingsideMoved = True

        # En passant move
        if move.isEnPassantMove:
            self.board[move.startRow][move.endCol] = "--"  # Capturing the pawn

        # Castling move
        if hasattr(move, 'isCastleMove') and move.isCastleMove:
            # If castling kingside, move rook from h-file to f-file
            if move.endCol - move.startCol == 2:  # kingside
                self.board[move.endRow][move.endCol - 1] = self.board[move.endRow][move.endCol + 1]
                self.board[move.endRow][move.endCol + 1] = "--"
            # If castling queenside, move rook from a-file to d-file
            else:  # queenside
                self.board[move.endRow][move.endCol + 1] = self.board[move.endRow][move.endCol - 2]
                self.board[move.endRow][move.endCol - 2] = "--"

        # Update enPassantPossible
        # Pawn double move logic for en passant
        if self.board[move.endRow][move.endCol] == "wp" and move.startRow == 6 and move.endRow == 4:
            self.enPassantPossible = (5, move.startCol)
        elif self.board[move.endRow][move.endCol] == "bp" and move.startRow == 1 and move.endRow == 3:
            self.enPassantPossible = (2, move.startCol)
        else:
            self.enPassantPossible = ()

        self.moveLog.append(move)  # log the game
        # Update half-move clock for 50-move rule
        # Reset halfMoveClock to 0 if pawn moved or capture; otherwise increment
        if move.pieceCaptured != "--" or move.pieceMoved[1] == "P":
            self.halfMoveClock = 0
        else:
            self.halfMoveClock += 1
        # Update repetition tracking using boardStateCounter
        self.boardStateCounter[self.getBoardStateKey()] = self.boardStateCounter.get(self.getBoardStateKey(), 0) + 1
        self.whiteToMove = not self.whiteToMove  # swap players
        # update the king's location
        if move.pieceMoved == 'wK':
            self.whiteKingLocation = (move.endRow, move.endCol)
        elif move.pieceMoved == 'bK':
            self.blackKingLocation = (move.endRow, move.endCol)
        if move.isPawnPromotion:
            self.board[move.endRow][move.endCol] = move.pieceMoved[0] + move.promotionChoice
        
    """
    undo the last move made
    """
    def undoMove(self):
        if len(self.moveLog) != 0:  # make sure that there is a move to undo
            # restore enPassantPossible
            if self.enPassantPossibleLog:
                self.enPassantPossible = self.enPassantPossibleLog.pop()
            else:
                self.enPassantPossible = ()
            # Restore halfMoveClock from log
            if hasattr(self, 'halfMoveClockLog') and self.halfMoveClockLog:
                self.halfMoveClock = self.halfMoveClockLog.pop()
            move = self.moveLog.pop()
            self.board[move.startRow][move.startCol] = move.pieceMoved
            self.board[move.endRow][move.endCol] = move.pieceCaptured
            # Undo en passant move
            if move.isEnPassantMove:
                self.board[move.endRow][move.endCol] = "--"
                self.board[move.startRow][move.startCol] = move.pieceMoved
                capturedRow = move.endRow + 1 if move.pieceMoved[0] == 'w' else move.endRow - 1
                self.board[capturedRow][move.endCol] = move.pieceCaptured
            # restore Castling rights
            if move.pieceMoved == 'wK':
                self.wKingMoved = False
            elif move.pieceMoved == 'bK':
                self.bKingMoved = False
            elif move.pieceMoved == 'wR':
                if move.startRow == 7 and move.startCol == 0:
                    self.wRookQueensideMoved = False
                elif move.startRow == 7 and move.startCol == 7:
                    self.wRookKingsideMoved = False
            elif move.pieceMoved == 'bR':
                if move.startRow == 0 and move.startCol == 0:
                    self.bRookQueensideMoved = False
                elif move.startRow == 0 and move.startCol == 7:
                    self.bRookKingsideMoved = False
            # Undo castling move
            if hasattr(move, 'isCastleMove') and move.isCastleMove:
                # If undoing kingside castling, move rook back from f-file to h-file
                if move.endCol - move.startCol == 2:  # kingside
                    self.board[move.endRow][move.endCol + 1] = self.board[move.endRow][move.endCol - 1]
                    self.board[move.endRow][move.endCol - 1] = "--"
                # If undoing queenside castling, move rook back from d-file to a-file
                else:  # queenside
                    self.board[move.endRow][move.endCol - 2] = self.board[move.endRow][move.endCol + 1]
                    self.board[move.endRow][move.endCol + 1] = "--"
            self.whiteToMove = not self.whiteToMove  # switch turns back
            # Reverse positionCount for draw rules
            if self.boardHistory:
                board_string = self.boardHistory.pop()
                if self.positionCount.get(board_string):
                    self.positionCount[board_string] -= 1
                    if self.positionCount[board_string] <= 0:
                        del self.positionCount[board_string]
            # update the king's location
            if move.pieceMoved == 'wK':
                self.whiteKingLocation = (move.startRow, move.startCol)
            elif move.pieceMoved == 'bK':
                self.blackKingLocation = (move.startRow, move.startCol)
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
                            logger.debug("check detected? %s checks: %s", inCheck, checks)
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

        # Check for checkmate/stalemate/50-move/3-fold
        self.checkMate, self.staleMate, self.draw50, self.drawRepetition = self.checkForEndConditions(moves)
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
                        logger.debug("checks piece %s at %s in directions %s", pieceType, (endRow, endCol), d)
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
             logger.debug('Checking if white is in check')
             return self.squareUnderAttack(self.whiteKingLocation[0], self.whiteKingLocation[1])
        else:
            logger.debug("Check if black in check")
            return self.squareUnderAttack(self.blackKingLocation[0], self.blackKingLocation[1])
        
        
    '''
    determine if the enemy can attack the square r ,c 
    '''    
    def squareUnderAttack(self, r, c):
        if self.insideSquareUnderAttack:
            return False
        self.insideSquareUnderAttack = True

        originalTurn = self.whiteToMove
        self.whiteToMove = not originalTurn
        oppMoves = self.getAllPossibleMoves()

        self.whiteToMove = originalTurn
        self.insideSquareUnderAttack = False

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
                    if 0 <= r + moveAmount < 8:
                        if self.board[r + moveAmount][c + dc][0] == enemyColor:
                            moves.append(Move((r, c), (r + moveAmount, c + dc), self.board))
                        elif (r + moveAmount, c + dc) == self.enPassantPossible:
                            moves.append(Move((r, c), (r + moveAmount, c + dc), self.board, isEnPassantMove=True))
                    
    '''
    get all the rook moves for the rook located at row, col and add these moves to the list
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
    get all the Knight moves for the knight located at row, col and add these moves to the list
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
    get all the Bishop moves for the bishop located at row, col and add these moves to the list
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
    get all the Queen moves for the queen located at row, col and add these moves to the list
    '''
    def getQueenMoves(self, r, c, moves):
        self.getRookMoves(r, c ,moves)
        self.getBishopMoves(r , c, moves)
    
    '''
    get all the King moves for the king located at row, col and add these moves to the list
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
                        moves.append(Move((r, c), (endRow, endCol), self.board))
        # Add castling moves
        self.getCastleMoves(r, c, moves)

    def getCastleMoves(self, r, c, moves):
        if self.squareUnderAttack(r, c):
            return
        if self.whiteToMove:
            if self.whiteKingLocation != (7,4) or self.wKingMoved:
                return
            # kingside
            if not self.wRookKingsideMoved and self.board[7][5] == "--" and self.board[7][6] == "--":
                if not self.squareUnderAttack(7,5) and not self.squareUnderAttack(7,6):
                    if self.board[7][7] == 'wR':
                        moves.append(Move((7,4), (7,6), self.board, isCastleMove=True))
            if not self.wRookQueensideMoved and self.board[7][1] == "--" and self.board[7][2] == "--" and self.board[7][3] == "--":
                if not self.squareUnderAttack(7,2) and not self.squareUnderAttack(7 ,3):
                    if self.board[7][0] == 'wR':
                        moves.append(Move((7,4), (7,2), self.board, isCastleMove=True))
        else:
            if self.blackKingLocation != (0,4) or self.bKingMoved:
                return
            # kingside
            if not self.bRookKingsideMoved and self.board[0][5] == "--" and self.board[0][6] == "--":
                if not self.squareUnderAttack(0, 5) and not self.squareUnderAttack(0,6):
                    if self.board[0][7] == 'bR':
                        moves.append(Move((0, 4), (0, 6), self.board, isCastleMove=True))
            if not self.bRookQueensideMoved and self.board[0][1] == "--" and self.board[0][2] == "--" and self.board[0][3] == "--":
                if not self.squareUnderAttack(0,2) and not self.squareUnderAttack(0,3):
                    if self.board[0][0] == 'bR':
                        moves.append(Move((0,4), (0,2), self.board, isCastleMove = True))
                    

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

    def checkForEndConditions(self, moves):
        """
        Check for checkmate, stalemate, 50-move draw, and repetition.
        Returns (checkmate, stalemate, draw50, drawRepetition)
        """
        if len(moves) == 0:
            if self.inCheck():
                return True, False, False, False  # Checkmate
            else:
                return False, True, False, False  # Stalemate

        if self.halfMoveClock >= 100:
            return False, False, True, False  # 50-move rule

        fen = self.getFEN()
        repetitions = self.positionCounts.get(fen, 0)
        if repetitions >= 3:
            return False, False, False, True  # 3-fold repetition

        return False, False, False, False  # No end condition

    def getFEN(self):
        """
        Generate a FEN string from the current board state.
        """
        fen_rows = []
        for row in self.board:
            fen_row = ""
            empty = 0
            for square in row:
                if square == "--":
                    empty += 1
                else:
                    if empty > 0:
                        fen_row += str(empty)
                        empty = 0
                    piece = square[1]
                    if square[0] == 'w':
                        fen_row += piece.upper()
                    else:
                        fen_row += piece.lower()
            if empty > 0:
                fen_row += str(empty)
            fen_rows.append(fen_row)
        fen_board = "/".join(fen_rows)
        fen_turn = 'w' if self.whiteToMove else 'b'
        return f"{fen_board} {fen_turn}"

class Move():
    # maps keys in values
    # key : value
    ranksToRows = {"1" : 7, "2": 6, "3": 5, "4" : 4,
                   "5" : 3, "6" : 2, "7" : 1, "8" : 0}
    rowsToRanks = {v : k for k, v in ranksToRows.items()}
    filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3,
                   "e": 4, "f": 5, "g":6, "h": 7}
    colsToFiles = {v: k for k, v in filesToCols.items()}

    def __init__(self, startSq, endSq, board, isCastleMove=False, isEnPassantMove=False):
        self.startRow = startSq[0]
        self.startCol = startSq[1]
        self.endRow = endSq[0]
        self.endCol = endSq[1]
        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]
        self.isEnPassantMove = isEnPassantMove
        self.enPassantPossible = ()
        if self.isEnPassantMove:
            self.pieceCaptured = 'bp' if self.pieceMoved == 'wp' else 'wp'

        # Pawn promotion
        self.isPawnPromotion = False
        if self.pieceMoved[1] == 'p':
            if (self.pieceMoved[0] == 'w' and self.endRow == 0) or \
               (self.pieceMoved[0] == 'b' and self.endRow == 7):
                self.isPawnPromotion = True
        self.moveID = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol
        self.promotionChoice = 'Q'  # default promotion
        self.isCastleMove = isCastleMove
    '''
    overriding the equals method
    '''
    
    def __eq__(self, other):
        if isinstance(other,Move):
            return self.startRow == other.startRow and self.startCol == other.startCol and \
                   self.endRow == other.endRow and self.endCol == other.endCol and \
                   self.pieceMoved == other.pieceMoved and self.isEnPassantMove == other.isEnPassantMove
        return False
            
         
         
    def getChessNotation(self):
        # you can add to make this like real chess notation
        return self.getRankFile(self.startRow, self.startCol) + self.getRankFile(self.endRow, self.endCol)
        
    def getRankFile(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]


    def checkForEndConditions(self, moves):
        if len(moves) == 0:
            if self.inCheck():
                return True, False, False, False  # checkmate
            else:
                return False, True, False, False  # stalemate
        if self.halfMoveClock >= 100:
            return False, False, True, False  # 50-move draw
        if self.boardStateCounter.get(self.getBoardStateKey(), 0) >= 3:
            return False, False, False, True  # 3-fold repetition
        return False, False, False, False  # game not ended

    def getBoardStateKey(self):
        return str(self.board) + str(self.whiteToMove)

    def hashBoard(self):
        return str(self.board) + str(self.whiteToMove)

    def isDraw(self):
        # 50-move rule
        if self.halfMoveClock >= 100:
            return True
        # Threefold repetition
        if any(count >= 3 for count in self.positionCount.values()):
            return True
        return False
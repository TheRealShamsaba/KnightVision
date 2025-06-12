"""
This is our main driver file. it will be responsible for handling user input and displaying current GameState Object
"""

import pygame as p

import chessEngine

import logging
logger = logging.getLogger(__name__)


WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT // DIMENSION
MAX_FPS = 15
IMAGES = {}

"""
initialize a global dic of images. this will be called exactly once in the main
"""

def Load_Images():
    pieces = ['wp', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bp', 'bR', 'bN', 'bB', 'bQ', 'bK']
    for piece in pieces:
        IMAGES[piece] = p.transform.scale(p.image.load("images/" + piece + ".png"), (SQ_SIZE, SQ_SIZE))
    #Note: we can access an image by sayin 'IMAGES['wp'] 
    
def main():
    """
    the main driver for our code. this will handle user input and updating the graphics
    """
    p.init()
    screen = p.display.set_mode((WIDTH, HEIGHT))
    clock = p.time.Clock()
    screen.fill(p.Color("white"))
    gs = chessEngine.GameState()

    validMoves = gs.getValidMoves()  # Make sure the method name is correctly spelled if necessary
    logger.debug("valid Moves:")
    for move in validMoves:
        logger.debug(move.getChessNotation())
    moveMade = False # flag variable for when move is made
    Load_Images() # only once before while loop
    running = True
    sqSelected = () # no square is selected, keep track of the last click of the user (tuple: (row , col))
    playerClicks = [] # keeps track of the player clicks (two tuple)

    while running:
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            # mouse handler
            elif e.type == p.MOUSEBUTTONDOWN:
                location = p.mouse.get_pos() # (x,y) location of mouse
                col = location[0]//SQ_SIZE
                row = location[1]//SQ_SIZE
                if sqSelected  == (row , col):
                    sqSelected = () #deselect
                    playerClicks = []
                else:
                    sqSelected = (row , col)
                    playerClicks.append(sqSelected) # appended for both 1st and 2nd clicks
                if len(playerClicks) == 2:  # after the second click
                    moveMade = False
                    for mv in validMoves:
                        if mv.startRow == playerClicks[0][0] and mv.startCol == playerClicks[0][1] and \
                           mv.endRow == playerClicks[1][0] and mv.endCol == playerClicks[1][1]:
                            piece = gs.board[mv.startRow][mv.startCol]
                            if (gs.whiteToMove and piece[0] == "w") or (not gs.whiteToMove and piece[0] == 'b'):
                                logger.debug("Attempting move: %s", mv.getChessNotation())
                                gs.makeMove(mv)
                                if mv.getChessNotation() == "e1c1":  # White queenside castling
                                    gs.board[7][0] = "--"
                                    gs.board[7][3] = "wR"
                                validMoves = gs.getValidMoves()
                                moveMade = True
                            break
                    if moveMade:
                        sqSelected = ()
                        playerClicks = []
                    else:
                        logger.debug("move not valid")
                        playerClicks = [sqSelected]
            # key handler
            elif e.type == p.KEYDOWN:
                if e.key == p.K_z: # undo when Z is pressed
                    gs.undoMove()
                    moveMade = True

        if moveMade:
            validMoves = gs.getValidMoves()
            moveMade = False

        drawGameState(screen , gs, validMoves, sqSelected)
        clock.tick(MAX_FPS)
        p.display.flip()
"""
responsible for all the graphics within a current game state
"""

def drawGameState(screen, gs, validMoves, sqSelected):
    drawBoard(screen)
    highlightSquares(screen, gs, validMoves, sqSelected)
    drawPieces(screen, gs.board)
    
    
def drawBoard(screen):
    colors = [p.Color("white") , p.Color("gray")]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[((r+c) %2) ]
            p.draw.rect(screen, color, p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE , SQ_SIZE))
            
            
    '''
    draw the pices on the board useing current GameState.board
    '''
def drawPieces(screen , board):
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            piece = board[r][c]
            if piece != "--" : # not empty square
                screen.blit(IMAGES[piece], p.Rect(c*SQ_SIZE, r*SQ_SIZE, SQ_SIZE, SQ_SIZE))


# highlight squares for selected piece and valid moves
def highlightSquares(screen, gs, validMoves, sqSelected):
    if sqSelected != ():
        r, c = sqSelected
        if gs.board[r][c][0] == ('w' if gs.whiteToMove else 'b'):
            s = p.Surface((SQ_SIZE, SQ_SIZE))
            s.set_alpha(100)
            s.fill(p.Color('blue'))
            screen.blit(s, (c * SQ_SIZE, r * SQ_SIZE))
            s.fill(p.Color('yellow'))
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (move.endCol * SQ_SIZE, move.endRow * SQ_SIZE))


if __name__ == "__main__":
    main()
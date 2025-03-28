"""
This is our main driver file. it will be responsible for handling user input and displaying current GameState Object
"""

import pygame as p 
import chessEngine


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
    
    """
    the main driver for our code. this will handel user input and updating the graphics
    """
    
def main():
        p.init()
        screen = p.display.set_mode((WIDTH, HEIGHT))
        clock = p.time.Clock()
        screen.fill(p.Color("white"))
        gs = chessEngine.GameState()
        Load_Images() #only once before while loop
        running = True
        sqSelected = () # no squre  is selected, keep track of the last click of the user (tuple: (row , col))
        playerClicks = [] # keeps track of the player clicks (two tuple)
        while running:
            for e in p.event.get():
                if e.type == p.QUIT:
                    running = False
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
                    if len(playerClicks) ==2 : # after the second click
                        move = chessEngine.Move(playerClicks[0], playerClicks[1], gs.board)
                        print(move.getChessNotaion())
                        gs.makeMove(move)
                        sqSelected = () #reset the user clicks
                        playerClicks = []
                    
            drawGameState(screen , gs)
            clock.tick(MAX_FPS)
            p.display.flip()
"""
responsible for all the graphics within a current game state
"""

def drawGameState(screen, gs):
    drawBoard(screen) #draw squres on board
    #add in piece gihlighting or move suggestions (later)
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


if __name__ == "__main__":
    main()
                
                
        
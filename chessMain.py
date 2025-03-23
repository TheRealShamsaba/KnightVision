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
    pieces = ['wp' , 'wR' , 'wN' , 'wK' , 'wQ' , 'bp' , 'bR' , 'bK' , 'bB' , 'bK' , 'bQ']
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
        while running:
            for e in p.event.get():
                if e.type == p.QUIT:
                    running = False
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
    pass


if __name__ == "__main__":
    main()
                
                
        
"""
This is our main driver file. it will be responsible for handling user input and displaying current GameState Object
"""

import pygame as p 
from chessBasic import chessEngine

WIDTH = HEIGHT = 512
DIMENSION = 8
SQ_SIZE = HEIGHT 
MAX_FPS = 15
IMAGES = {}

"""
initialize a global dic of images. this will be called exactly once in the main
"""

def Load_Images():
    pieces = ['wp' , 'wR' , 'wN' , 'wK' , 'wQ' , 'bp' , 'bR' , 'bK' , 'bB' , 'bK' , 'bQ']
    for piece in pieces:
        IMAGES["wp"] = p.image.load("images/" + piece + ".png")
    #Note: we can access an image by sayin 'IMAGES['wp'] 
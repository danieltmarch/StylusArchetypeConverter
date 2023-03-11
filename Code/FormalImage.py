#handles rendering the formal image of a certain character, relatively simple.
#alternatively we can use tkinter to make images, but the only difference would be image size and position
#which shouldn't really matter

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np

#Heavily based on https://stackoverflow.com/a/27753869/190597 (jsheperd)
#renders as 1 = black, 0 = white
def renderChar(char, path='../Fonts/msyh.ttc', fontsize=1000, show=False):
    font = ImageFont.truetype(path, fontsize) 
    w, h = font.getsize(char)
    h *= 2
    image = Image.new('L', (w, h), 1)  
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), char, font=font)
    arr = np.asarray(image)
    arr = np.where(arr, 0, 1)
    arr = arr[(arr != 0).any(axis=1)]
    
    #to save some pain we'll flip the y axis so it reads like a normal graph
    arr = np.flip(arr, axis=0)
    
    #we need some padding on the edges, 20% to height and width each
    height = len(arr)
    width = len(arr[0])
    hPad = height//10 #10% of height, int division
    wPad = width//10 #10 of width, int division
     
    paddedImg = np.zeros( (height+2*hPad, width+2*wPad) )
    paddedImg[ hPad:hPad+height , wPad:wPad+width ] = arr
    
    if(show):
        plt.imshow(paddedImg, origin='lower')    
    return paddedImg
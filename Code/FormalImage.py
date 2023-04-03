#handles rendering the formal image of a certain character, relatively simple.
#alternatively we can use tkinter to make images, but the only difference would be image size and position
#which shouldn't really matter

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import cv2 #can be installed with pip install opencv-python

#Heavily based on https://stackoverflow.com/a/27753869/190597 (jsheperd)
#renders as 1 = black, 0 = white
#size is the final width/height of the image
def renderChar(char, path='../Fonts/msyh.ttc', size=1000, fontsize=1000, pad=0.1, show=False):
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
    #arr = np.flip(arr, axis=0) #not currently used
    
    #we need some padding on the edges, 20% to height and width each
    height = len(arr)
    width = len(arr[0])
    hPad = int(height*pad/2) #half pad on each side
    wPad = int(width*pad/2)
     
    paddedImg = np.zeros( (height+2*hPad, width+2*wPad) )
    paddedImg[ hPad:hPad+height , wPad:wPad+width ] = arr
    
    currDim = np.array(np.shape(paddedImg))
    maxDim = max(currDim)
    imgSquare = np.zeros((maxDim,maxDim))
    offset = (maxDim - currDim)//2 #so image is centered
    imgSquare[offset[0]:offset[0]+currDim[0], offset[1]:offset[1]+currDim[1]] = paddedImg

    img = cv2.resize(imgSquare, dsize=(size, size), interpolation=cv2.INTER_NEAREST) #nearest neighbor interpolation
    
    if(show):
        plt.imshow(img, cmap='gray', vmin=0, vmax=20)
    return img
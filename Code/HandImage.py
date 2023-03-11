#handles rendering the hanzi hand written image of a certain character, using the hanzi writer app data.

import numpy as np
import json
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

#char data structure
def getHanziChar(character):
    fileName = '../Hanzi Data/' + character + '.json'
    with open(fileName, 'r') as f:
        charData = json.load(f)
    return charData

def getStrokeData(character):
    charData = getHanziChar(character)
    strokeData = [] #each entry is a stroke (theoretically in the right order)
    for stroke in charData['medians']:
        pointData = [[],[]] #[xPoints, yPoints]
        for point in stroke:
            pointData[0].append(point[0])
            pointData[1].append(point[1])
        strokeData.append(pointData)
    return strokeData
    
def plotChar(character):
    strokeData = getStrokeData(character)
    colors=["red","darkorange","green","dodgerblue","blue","purple"]
    fig, ax = plt.subplots(figsize=(16,16))
    
    for i in range(len(strokeData)):
        ax.plot(strokeData[i][0], strokeData[i][1], color=colors[i%len(colors)], linewidth=3.5)
    
    #make the border as tight as possible
    xPoints = []
    yPoints = []
    for stroke in strokeData:
        xPoints.extend(stroke[0])
        yPoints.extend(stroke[1])
    
    xRange = [min(xPoints), max(xPoints)]
    yRange = [min(yPoints), max(yPoints)]
    wPad = (xRange[1] - xRange[0])//10
    hPad = (yRange[1] - yRange[0])//10
    
    ax.set_xlim([xRange[0]-wPad, xRange[1]+wPad]) #10% padding on both wides
    ax.set_ylim([yRange[0]-hPad, yRange[1]+hPad]) #10% padding
    
    return fig, ax

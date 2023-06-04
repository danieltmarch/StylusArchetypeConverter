import numpy as np
import matplotlib.pyplot as plt

import HandImage as hi
import StrokeFit as sf
from StrokeDef import StrokeType #needed for loading object properly
import StrokeDef as sd
from FitDef import FitData
import FitDef as fd
import time
import Anchors as an

def fitChar(character, size=1000, printOut=False):
    strokeData = hi.getStrokeData(character)
    if(strokeData is None):
        return False #failed
    if(printOut):
        figChar, axChar = hi.plotChar(character)
    strokeDict = sd.loadStrokeDict()
    fits = sf.getFits(strokeData, strokeDict)
    if(printOut):
        figFit, axFit = sf.graphFits(strokeData, strokeDict, fits)

    handDims = np.array([1000, 1000]) #this appears to be default always
    arialDims = np.array([size,size])
    arialFits = sf.mapFits(fits, handDims, arialDims, pad=.1) #.1 padding on both sides

    fitData = FitData()
    fitData.set(character, arialFits, arialDims)
    
    #remaps based on anchors and arial geometry
    sm = an.StrokeMapper()
    fitData = sm.remapFits(character, fitData=fitData)
    
    fd.saveFits(fitData)
    return True

def processRange(start, end, size=100, overwrite=True): #over is if we write over existing files
    processCount = 0
    startTime = time.time()
    for curr in range(ord(start), ord(end)+1):
        currChar = chr(curr)
        if( (not overwrite) and fd.fileExists(currChar)): #file exists, skip so we don't overwrite
            print(f"File for {currChar} exists! Skipping...")
            continue
        if(fitChar(currChar)):
            processCount += 1
            print(f"{currChar} fitted and saved!")
        else:
            print(f"No hanzi writer data found for {currChar}! Skipping...")
    duration = time.time() - startTime
    avrTime = 0 if processCount==0 else duration/processCount #avoid divide by zero error
    print(f"{processCount} characters processed in {round(duration,4)} seconds! Average sec per char: {round(avrTime,4)}.")
    return processCount
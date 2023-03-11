#handles the data for formal and handwritten strokes

import bezier
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------

#each stroke is accessed by a key (e.g. 'c0') and is a list of [x,y] coords
def saveHandStrokes(dictionary):
    np.save('HandStrokes.npy', dictionary) 
def saveFormalStrokes(dictionary):
    np.save('FormalStrokes.npy', dictionary) 
def loadHandStrokes():
    return np.load('HandStrokes.npy', allow_pickle=True).item()
def loadFormalStrokes():
    return np.load('FormalStrokes.npy', allow_pickle=True).item()

# ------------------------------

#a couple classes to make using this file easier to read/use.
class BoundingBox:
    def __init__(self, rawData):
        self.minX = rawData[0][0]
        self.maxX = rawData[0][1]
        self.minY = rawData[1][0]
        self.maxY = rawData[1][1]
class StrokeMatch: #a matching stroke
    def __init__(self, strokeType, style, boundingBox):
        self.type = strokeType
        if(style == 'hand'):
            self.style = 'hand'
        else:
            self.style = 'formal'
        self.bounds = boundingBox

# ------------------------------
        
def findStrokeBounds(hanziStroke): #from hand written data of hanzi app
    #[[minX,maxX], [minY, maxY]
    b = BoundingBox([ [min(hanziStroke[0]), max(hanziStroke[0])], 
                      [min(hanziStroke[1]), max(hanziStroke[1])] ])
    #avoid having to crazy an aspect ratio
    #if window is really narrow miden this, as it is clearly a horizontal or vertical line
    if((b.maxX - b.minX) < .25*(b.maxY - b.minY)): #4:1 aspect ratio, widen the x dir, make at least #1:2
        middle = int((b.maxX + b.minX)/2)
        minDist = int(.333*(.5*(b.maxY - b.minY))) #how much to widen from the middle, .5 for 1:2
        b.maxX = middle+minDist
        b.minX = middle-minDist
    elif((b.maxY - b.minY) < .25*(b.maxX - b.minX)): #4:1 aspect ratio, widen the y dir, make at least #4:3
        middle = int((b.maxY + b.minY)/2)
        minDist = int(.333*(.5*(b.maxX - b.minX))) #how much to widen from the middle, .5 for 1:2
        b.maxY = middle+minDist
        b.minY = middle-minDist
    return b

# ------------------------------

def fitStrokes(hanziCharData):
    fits = []
    for hanziStroke in hanziCharData:
        fits.append(findBestStroke(np.array(hanziStroke)))
    return fits
#from hand written data of hanzi app, returns a StrokeMatch type
def findBestStroke(hanziStroke):
    strokeDict = loadHandStrokes()
    bounds = findStrokeBounds(hanziStroke)
    
    bestType = 'c0' #default val
    bestError = 10e20 #arbitrary large value
    for strokeType in strokeDict:
        strokeData = np.array(strokeDict[strokeType])
        #resize the stroke to the bounding box
        strokeData[:,:,0] = bounds.minX + (strokeData[:,:,0]*(bounds.maxX-bounds.minX))
        strokeData[:,:,1] = bounds.minY + (strokeData[:,:,1]*(bounds.maxY-bounds.minY))
        
        distError = getLineErrors(hanziStroke, strokeData, resol=100)
        if(distError < bestError): #new best
            bestError = distError
            bestType = strokeType
    
    strokeData = np.array(strokeDict[bestType])
    strokeData[:,:,0] = bounds.minX + (strokeData[:,:,0]*(bounds.maxX-bounds.minX))
    strokeData[:,:,1] = bounds.minY + (strokeData[:,:,1]*(bounds.maxY-bounds.minY))
    return StrokeMatch(bestType, 'hand', bounds) #return the stroke format

# ------------------------------

def translateFits(fits, handShape, formalShape):
    newFits = []
    for fit in fits:
        newFits.append(translateStroke(fit, handShape, formalShape))
    return newFits

def boundMap(oldPos, oldMin, oldMax, newMin, newMax):
    percent = float(oldPos-oldMin)/(oldMax-oldMin)
    return int(newMin + (percent*(newMax - newMin)) )
        
#we need the shape of each image and the stroke match for hand written
def translateStroke(handStrokeMatch, hiBound, fiBound):
    handDict = loadHandStrokes()
    formalDict = loadFormalStrokes()
    
    hiBound = BoundingBox(hiBound)
    fiBound = BoundingBox(fiBound)
    
    oldBounds = handStrokeMatch.bounds
    newBounds = BoundingBox([[0,0],[0,0]]) #we'll fix these values shortly
    #simple mapping based on the image size
    newBounds.minX = boundMap(oldBounds.minX, hiBound.minX, hiBound.maxX, fiBound.minX, fiBound.maxX)
    newBounds.maxX = boundMap(oldBounds.maxX, hiBound.minX, hiBound.maxX, fiBound.minX, fiBound.maxX)
    newBounds.minY = boundMap(oldBounds.minY, hiBound.minY, hiBound.maxY, fiBound.minY, fiBound.maxY)
    newBounds.maxY = boundMap(oldBounds.maxY, hiBound.minY, hiBound.maxY, fiBound.minY, fiBound.maxY)
    
    return StrokeMatch(handStrokeMatch.type, 'formal', newBounds)        

# ------------------------------

def plotHandFits(fits, ax=False, drawBox=True): #draw on some plot the fits for the strokes
    if(not ax): #we weren't given an existing plot, make a new one
        fig, ax = plt.subplots(figsize=(16,16))
    strokeDict = loadHandStrokes()
    
    evalPoints = np.linspace(0.0, 1.0, 100) #start, end, resol, parametric t values
    for fit in fits:
        stroke = np.array(strokeDict[fit.type])
        b = fit.bounds
        #resize and shift appropriately
        stroke[:,:,0] = b.minX + (stroke[:,:,0]*(b.maxX-b.minX))
        stroke[:,:,1] = b.minY + (stroke[:,:,1]*(b.maxY-b.minY))
        
        for curve in stroke: #control points, 
            nodes = np.array(curve).transpose()
            curve = bezier.Curve(nodes, degree=len(curve)-1)
            curvePoints = np.array(curve.evaluate_multi(evalPoints).transpose().tolist()) #[[x,y],...]
            ax.plot(curvePoints[:,0], curvePoints[:,1], color="black")
            
        if(drawBox): #plot the border box too
            ax.plot([b.minX,b.minX],[b.minY,b.maxY], linestyle="dashed", color="blue")
            ax.plot([b.maxX,b.maxX],[b.minY,b.maxY], linestyle="dashed", color="blue")
            ax.plot([b.minX,b.maxX],[b.maxY,b.maxY], linestyle="dashed", color="blue")
            ax.plot([b.minX,b.maxX],[b.minY,b.minY], linestyle="dashed", color="blue")
    return ax

#draw the formal img with the fits, keep in mind this is a matrix not a plot
#0 = white, 1 = black, 2 = stroke, 3 = bounding box
def drawFormalFits(fits, img, drawBox=True):
    strokeDict = loadFormalStrokes()
    newImg = img.copy()
    evalPoints = np.linspace(0.0, 1.0, 500) #start, end, resol, parametric t values
    for fit in fits:
        stroke = np.array(strokeDict[fit.type])
        b = fit.bounds
        #resize and shift appropriately
        stroke[:,:,0] = b.minX + (stroke[:,:,0]*(b.maxX-b.minX))
        stroke[:,:,1] = b.minY + (stroke[:,:,1]*(b.maxY-b.minY))
        
        for curve in stroke: #control points, 
            nodes = np.array(curve).transpose()
            curve = bezier.Curve(nodes, degree=len(curve)-1)
            curvePoints = np.array(curve.evaluate_multi(evalPoints).transpose().tolist()) #[[x,y],...]
            for point in curvePoints:
                newImg[int(point[1]),int(point[0])] = 20
            
        if(drawBox): #plot the border box too
            newImg[b.minY:b.maxY, b.minX] = 10
            newImg[b.minY:b.maxY, b.maxX] = 10
            newImg[b.minY, b.minX:b.maxX] = 10
            newImg[b.maxY, b.minX:b.maxX] = 10
        
    return newImg

# ------------------------------

#### ---- from here downwards is all the logic for calculating stroke errors between lines

def pointDist(coord1, coord2): #coord = [x,y]
    return np.sqrt((coord2[1] - coord1[1])**2 + (coord2[0] - coord1[0])**2)

#stroke = [[x,y],[x,y] ... [x,y]], make new points on these segments that are equally spaced
def getPoints(stroke, count): #get certain amount of evenly spaced points from hanzi line data
    stroke = np.array(stroke)
    totalLineDistance = 0
    incrDist = [0]
    for i in range(1, len(stroke)):
        totalLineDistance += pointDist(stroke[i-1], stroke[i])
        incrDist.append(totalLineDistance)
    segDist = (totalLineDistance / (count-1)) #-1 since we can use orgin and end as points on the line
    
    samplePoints = [stroke[0]] #start with first point
    currSeg = 0 #currSeg 0 is the first line on the graph, len(stroke)-2 is the last
    for i in range(1,count-1): #we already have the first point in, don't do the final point
        goalDist = segDist*i
        while(currSeg + 1 < len(incrDist) and goalDist > incrDist[currSeg+1]): #move on to the correct segment
            currSeg += 1
        if(currSeg+1 >= len(incrDist)):
            break
        goalDist -= incrDist[currSeg] #how much further we need to go for this segment
        #figure out what frac of this segment we need to move to get our point
        fracDist = goalDist/(incrDist[currSeg+1]-incrDist[currSeg])
        samplePoints.append(stroke[currSeg] + fracDist*(stroke[currSeg+1] - stroke[currSeg]))
    samplePoints.append(stroke[-1]) #final sample point
    return np.array(samplePoints)

def getPointsHanzi(stroke, count): #get certain amount of evenly spaced points from hanzi line data
    points = np.array(stroke).transpose() #[[x,y],[x,y], ...]
    return getPoints(points, count) #we've simplified our curve to many segments

#bezier curves are harder so just turn them into a ton of line segments and then run general get point func
#note a stroke should be continuous (it shouldn't have a grap)
def getPointsBezier(stroke, count, resol=100): #get certain amount of evenly spaced points from the formal
    evalPoints = np.linspace(0.0, 1.0, resol) #start, end, resol, parametric t values
    allLinePoints = [] #each begin/end of a line segment
    for curve in stroke:
        nodes = np.array(curve).transpose()
        curve = bezier.Curve(nodes, degree=len(curve)-1)
        curvePoints = curve.evaluate_multi(evalPoints).transpose().tolist() #[[x,y], ...]
        allLinePoints.extend(curvePoints)
        
    return getPoints(np.array(allLinePoints), count) #we've simplified our curve to many segments

#get the error for the stroke
#each hanzi point nearest distance to formal stroke all summed
#and vice versa is the total error
def getLineErrors(hanziStroke, formalStroke, resol=200):
    hanziSample = getPointsHanzi(hanziStroke, resol)
    bezierSample = getPointsBezier(formalStroke, resol)
    
    totalError = 0
    for point in hanziSample:
        totalError += getNearestDist(point, bezierSample)
    for point in bezierSample:
        totalError += getNearestDist(point, hanziSample)
    return totalError
    
#find the nearest distance from the point to a line point
def getNearestDist(point, linePoints):
    minDist = 10e10 #arbitrary large value
    for p in linePoints:
        dist = pointDist(point, p)
        if(dist < minDist):
            minDist = dist
    return minDist
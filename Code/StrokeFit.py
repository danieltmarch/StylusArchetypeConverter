#contains functions for fitting hanzi stroke data with our native stroke definitions, also does the mapping from hanzi coords to formal image
import numpy as np
import bezier
import matplotlib.pyplot as plt

#functions for fitting hanzi strokes against our stroke definitions.

#a fit consists of a stroke name and bound box,
class Fit:
    def __init__(self, name, xRange, yRange):
        self.name = name
        self.x = np.array(xRange)
        self.y = np.array(yRange)
        self.group = 1
        
    def __str__(self):
        return (f"Stroke type {self.name}. x bounds: {self.x}, y bounds: {self.y}.")
    
#fit each stroke using the strokeDef dictionary {Key: name, Value: StrokeDef}
def getFits(strokes, strokeDefs, resol=50):
    fits = [] #list of fit types, order matches strokes in hanzi strokeData
    evalPoints = np.linspace(0.0, 1.0, resol) #used for calculating error. start, end, resol, parametric t values
    for stroke in strokes:
        stroke = np.transpose(stroke) #in easier format for error function to work with, list of [x,y] coords now
        xRange = [min(stroke[:,0]), max(stroke[:,0])]
        yRange = [min(stroke[:,1]), max(stroke[:,1])]
        xDiff = xRange[1] - xRange[0]
        yDiff = yRange[1] - yRange[0]
        
        #if window is really narrow, widen it, as it is must be a horizontal or vertical line
        if(xDiff < .25*yDiff): #4:1 aspect ratio, widen the x dir, make at least #4:3
            xRange[1] += .25*yDiff
            xRange[0] -= .25*yDiff
        if(yDiff < .25*xDiff): #4:1 aspect ratio, widen the y dir, make at least #4:3
            yRange[1] += .25*xDiff 
            yRange[0] -= .25*xDiff
            
        bestStroke = None #default val
        bestType = "None"
        bestError = 10e20 #arbitrary large value
        for strokeType in strokeDefs.values(): #since values are strokeDefs, keys are just the name
            strokeData = mapStroke(strokeType.hanzi, xRange, yRange) #strokeType bezier curve data
            #now we compare manualStroke and the strokeData
            distError = getLineErrors(stroke, strokeData, resol=100)
            if(distError < bestError): #new best, a threshhold could be set here too
                bestError = distError
                bestStroke = strokeData
                bestType = strokeType.name
        fits.append(Fit(bestType, xRange, yRange)) #add to fits
    return fits

#map a given stroke originally in a 0 to 1 bounding box, to a new xRange and yRange.
def mapStroke(strokeData, xR, yR):
    strokeData = np.array(strokeData)
    strokeData[:,:,0] = xR[0] + (strokeData[:,:,0]*(xR[1]-xR[0]))
    strokeData[:,:,1] = yR[0] + (strokeData[:,:,1]*(yR[1]-yR[0]))
    return strokeData

#graph using the fits, hanzi strokeData (dictionary), and fits (list)
def graphFits(strokes, strokeDefs, fits, resol=25):
    fig, ax = plt.subplots(figsize=(16,16))
    ax.invert_yaxis()
    for stroke in strokes: #graph hand written strokes
        ax.plot(stroke[0], stroke[1], color="red", linewidth=4, linestyle="dotted") #flips y axis
    
    evalPoints = np.linspace(0.0, 1.0, resol) #determines how accurate bezier curves are drawn
    for fit in fits: #plot bounding box and bezier curves
        arr2 = np.ones(2) # just [1,1], makes these next lines a bit more readable
        ax.plot( fit.x[0]*arr2, fit.y, linestyle="dashed", color="blue", linewidth=3) #plot bounding box
        ax.plot( fit.x[1]*arr2, fit.y, linestyle="dashed", color="blue", linewidth=3)
        ax.plot( fit.x, fit.y[0]*arr2, linestyle="dashed", color="blue", linewidth=3)
        ax.plot( fit.x, fit.y[1]*arr2, linestyle="dashed", color="blue", linewidth=3)
        
        for curve in mapStroke(strokeDefs[fit.name].hanzi, fit.x, fit.y): #control points, 
            nodes = np.array(curve).transpose()
            curve = bezier.Curve(nodes, degree=len(curve)-1)
            curvePoints = np.array(curve.evaluate_multi(evalPoints).transpose().tolist()) #[[x,y]...]
            ax.plot(curvePoints[:,0], curvePoints[:,1], color="black", linewidth=3)
    return fig, ax

#function for calculating error for matching strokes
def pointDist(coord1, coord2): #coord = [x,y]
    return np.sqrt((coord2[1] - coord1[1])**2 + (coord2[0] - coord1[0])**2)

def getPointsHanzi(stroke, count): #get certain amount of evenly spaced points from hanzi line data
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
        while(goalDist > incrDist[currSeg+1]): #move on to the correct segment
            currSeg += 1
        goalDist -= incrDist[currSeg] #how much further we need to go for this segment
        #figure out what frac of this segment we need to move to get our point
        fracDist = goalDist/(incrDist[currSeg+1]-incrDist[currSeg])
        samplePoints.append(stroke[currSeg] + fracDist*(stroke[currSeg+1] - stroke[currSeg]))
    samplePoints.append(stroke[-1]) #final sample point
    return np.array(samplePoints)

#bezier curves are harder so just turn them into a ton of line segments and then run the hanzi version
#note a stroke should be continuous (it shouldn't have a grap)
def getPointsBezier(stroke, count, resol=50): #get certain amount of evenly spaced points from the formal
    evalPoints = np.linspace(0.0, 1.0, resol) #start, end, resol, parametric t values
    allLinePoints = [] #each begin/end of a line segment
    for curve in stroke:
        nodes = np.array(curve).transpose()
        curve = bezier.Curve(nodes, degree=len(curve)-1)
        curvePoints = np.array(curve.evaluate_multi(evalPoints).transpose().tolist()) #[[x,y]...]
        allLinePoints.extend(curvePoints)
    return getPointsHanzi(allLinePoints, count) #we've simplified our curve to many segments

#get the error for the stroke
#each hanzi point nearest distance to formal stroke all summed
#and vice versa is the total error
def getLineErrors(hanziStroke, strokeDef, resol=200):
    hanziSample = getPointsHanzi(hanziStroke, resol)
    bezierSample = getPointsBezier(strokeDef, resol, resol=100)
    
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

#mapping function to map from hanzi coords to arial coords

def mapCoord(coord, handDim, arialDim, pad): #e.g. x coord 400, when original width is 800, and new width is 1000, maps to 500, pad is a percent of the arialDim
    pad = pad/2
    return arialDim*pad + (arialDim*(1-pad))*(coord/handDim)
#fits is a list, handDims and arialDims are a list like: [width, height]
def mapFits(fits, handDims, arialDims, pad): #pad is percent of the arial dims, should be same on both side
    arialFits = []
    for i in range(len(fits)): #map the bounding box coords
        arialFits.append(Fit(fits[i].name, fits[i].x, fits[i].y))
        arialFits[i].x[0] = round(mapCoord(arialFits[i].x[0], handDims[0], arialDims[0], pad)) #bounds need to be integers, so round
        arialFits[i].x[1] = round(mapCoord(arialFits[i].x[1], handDims[0], arialDims[0], pad))
        arialFits[i].y[0] = round(mapCoord(arialFits[i].y[0], handDims[1], arialDims[1], pad))
        arialFits[i].y[1] = round(mapCoord(arialFits[i].y[1], handDims[1], arialDims[1], pad))
    return arialFits
import numpy as np
#back end libraries for loading/saving/modifying our data
import StrokeFit as sf
import FormalImage as fi
from FitDef import FitData
import FitDef as fd
from StrokeDef import StrokeType #needed for loading object properly
import StrokeDef as sd
import matplotlib.pyplot as plt
import bezier

class Corner:
    def __init__(self, coord, cType, angle):
        self.coord = np.array(coord)
        self.type = cType #-1 for inverted (3/4 char), 1 for normal (3/4 non char)
        self.angle = angle

#handles converting an img to the anchor points
class AnchorConverter:
    def __init__(self):
        
        #for getDiff
        self.DiffBorderDist=10 #borderDiff is how far to check away from each pixel
        
        #for get Corners
        self.diffThresh=.2 #ratio diff between white/black needed to be considered for a corner
        self.borderDiff=5 #how big a distance we look for a local minimum (smaller may mean more corners detected)
        self.cornerPrint=False #prints info about the lowest passed pixel threshhold
        
        #for growImg
        self.growDist = 3 #for the grown image how many pixels to grow the image
        
        #for getAnchors
        self.connectedSampleNum = 5 #how many samples to take to check if corners are connected
        self.closePercent = .14 #up to this % of the image span may allow corners to be considered to be in the same group
        self.closestPercent = .03 #corners this close won't be considered to be of the same corner
        
    #get and return the anchor points
    def get(self, img, getCorners=False):
        #get diff img, difference between nearby white/black neighbors
        diffImg = self._getDiff(img, borderDiff=self.DiffBorderDist)
        #get a list of corner coordinates, which we'll use to determine the anchors
        corners = self._getCorners(img, diffImg, diffThresh=self.diffThresh, borderDiff=self.borderDiff, printInfo=self.cornerPrint)
        #get a grown version of the image, used for checking if corners belong to the same anchor point
        #growImg = self._getGrown(img, growDist=self.growDist)
        
        cornersCopy = []
        for c in corners:
            cornersCopy.append( Corner(c.coord, c.type, c.angle) )
        
        anchors = self._getAnchors(img, cornersCopy) #we will be deleting corners copy
        
        if(getCorners):
            return anchors, corners
        return anchors

    #private functions ↓↓↓ ------------- (noted with an underscore in front of name

    #go through each pixel and count it's white and black neighbors, then each cell value is abs(blackCount - whiteCount)
    #borderDiff is how far to check away from each pixel
    def _getDiff(self, img, borderDiff=8):
        diffImg = np.zeros(np.shape(img))

        imgColor = img.copy() #black is set to 1 already
        imgColor[img == 0] = -1 #set white pixels to -1

        pad = 1 #ignore the border for this thickness
        totalPad = borderDiff+pad
        pad = 1 #ignore the border for this thickness
        #somewhat complicated use of masks to achieve fast calculation of the diffImg
        for xDiff in range(-borderDiff,1+borderDiff):
            for yDiff in range(-borderDiff,1+borderDiff):
                if(xDiff == 0 and yDiff == 0): #we don't count the center
                    continue
                diffImg[totalPad:-totalPad, totalPad:-totalPad] += imgColor[totalPad+yDiff:-totalPad+yDiff, totalPad+xDiff:-totalPad+xDiff]

        #now determine which pixels are on the border (black but touches a white pixel, corner doesn't count)
        borderImg = np.zeros(np.shape(img)) #zero is not a border
        borderImg[1:-1, 1:-1][(img[1:-1, 1:-1] == 1) & ( img[:-2, 1:-1] == 0)] = 1 #up direction
        borderImg[1:-1, 1:-1][(img[1:-1, 1:-1] == 1) & (img[2:, 1:-1] == 0)] = 1 #down direction
        borderImg[1:-1, 1:-1][(img[1:-1, 1:-1] == 1) & (img[1:-1, :-2] == 0)] = 1 #left direction
        borderImg[1:-1, 1:-1][(img[1:-1, 1:-1] == 1) & (img[1:-1, 2:] == 0)] = 1 #right direction

        diffImg[borderImg != 1] = 0 #reset any pixels that aren't a border
        diffImg /= (borderDiff*2)**2 - 1 #ratio of total pixels counted
        return np.abs(diffImg)

    #from the diff image find the pixels which are the "corners" of the character
    #diffThresh is the relative difference between black and white for a pixel to be considered as a corner (recommended: 25%)
    #borderDiff is how far to check away from each pixel for a better ratio
    def _getCorners(self, img, diffImg, diffThresh = .25, borderDiff=5, printInfo=True):
        maxImg = np.zeros(np.shape(diffImg))
        maxImg[diffImg > diffThresh] = 1 #anything above this difference is a canidate for being a corner

        #each corner should be simplified to one single pixel, the highest ratio pixel
        #therefore, we'll try to find the local maximums which will be our corners.
        pad = 1
        totalPad = borderDiff+pad
        centerMask = diffImg[totalPad:-totalPad, totalPad:-totalPad]
        for xDiff in range(-borderDiff,1+borderDiff): #similar method used for the getDiff function
            for yDiff in range(-borderDiff,1+borderDiff):
                if(xDiff == 0 and yDiff == 0): #we don't count the center
                    continue
                offsetMask = diffImg[totalPad+yDiff:-totalPad+yDiff, totalPad+xDiff:-totalPad+xDiff]
                maxImg[totalPad:-totalPad, totalPad:-totalPad][ centerMask < offsetMask ] = 0 #neighbor is higher value, reset to zero
        maxImg[maxImg != 0] = 1 #1's are where corners are located

        #there may be multiple very close pixels that tied, just give precedence to the lower y/x values to fix the ties.
        #maybe a better way to do this without using native loops
        for y in range(len(maxImg)):
            for x in range(len(maxImg[0])):
                if(maxImg[y,x] == 1):
                    maxImg[y-borderDiff:y+borderDiff, x-borderDiff:x+borderDiff] = 0 #set all nearby to 0
                    maxImg[y,x] = 1 #reset the center back to 1

        #print("Lowest ratio corner:", min(diffImg[maxImg == 1])) #lowest diff that passed the test
        #print("Highest ratio corner:", max(diffImg[maxImg == 1])) #highest diff that passed the test

        cornersRaw = np.array(np.column_stack(np.where(maxImg == 1))) #corners like: [ [corner1Y, corner1X], [corner2Y, corner2X], [corner3Y, corner3X] ... ]
        
        corners = []
        for c in cornersRaw:
            corners.append( Corner(c, self._getCornerType(img, c), self._getAngle(img, c)) )
        return corners

    def _getCornerType(self, img, coord, dist=8):
        subImg = img[coord[0]-dist:coord[0]+dist+1, coord[1]-dist:coord[1]+dist+1]
        if(np.sum(subImg == 1) > (dist*2 + 1)**2 / 2 ): #more black than white
            return -1 #inverted
        else:
            return 1 #normal corner
    
    def _getAngle(self, img, coord, dist=12): #dist is how far from the center point we check to determine the angle
        coord = np.array(coord)
        vect = np.zeros(2)
        for i in range(coord[0]-dist, coord[0]+dist+1):
            for j in range(coord[1]-dist, coord[1]+dist+1):
                dVect = np.array([i,j]) - coord
                if(dVect[0]**2 + dVect[1]**2 > dist**2):
                    continue #too far
                if(img[i,j] == 1): #black pixel
                    vect += dVect
        return np.angle(complex(vect[1], vect[0]), deg=True) #easy way to get angle by using complex coords
    
    #from the image, grow the black pixels
    #used for detecting if corners belong to the same anchor point
    def _getGrown(self, img, growDist=3):
        growImg = img.copy()

        pad = 1
        totalPad = growDist+pad
        #centerMask = img[totalPad:-totalPad, totalPad:-totalPad]
        for xDiff in range(-growDist,1+growDist): #similar method used for the getDiff function
            for yDiff in range(-growDist,1+growDist):
                if(xDiff == 0 and yDiff == 0): #we don't count the center
                    continue
                offsetMask = img[totalPad+yDiff:-totalPad+yDiff, totalPad+xDiff:-totalPad+xDiff]
                growImg[totalPad:-totalPad, totalPad:-totalPad][ offsetMask == 1 ] = 1 #set black since current offset is black
        return growImg
    
    
    def getIntersection(self, c1,c2):
        s1 = np.sin(np.radians(c1.angle))/np.cos(np.radians(c1.angle)) #numpy actually works fine and won't trigger div by 0 errors
        s2 = np.sin(np.radians(c2.angle))/np.cos(np.radians(c2.angle))

        b1 = c1.coord[0]-s1*c1.coord[1] #y-mx = b
        b2 = c2.coord[0]-s2*c2.coord[1] #y-mx = b

        x = (b2-b1)/(s1-s2)
        y = s1*x + b1
        return np.array([y,x])
    
    def getAngleDiff(self, a1,a2): #angles are in range -180 to 180
        angleDiff = abs(a1-a2) #0 to 360
        if(angleDiff > 180):
            return 360-angleDiff #bound between 0-180
        return angleDiff #else angle is already 0-180
    
    def _getAnchors(self, img, corners, angleTol = 15):
        anchors = []
        closestCorner = len(img)*self.closestPercent
        farthestCorner = len(img)*self.closePercent
        
        while(len(corners) != 0):
            currC = corners.pop(0) #look for a match for this corner
            nearby = []
            for c in corners:
                dist = np.sqrt(sum((currC.coord - c.coord)**2))
                if(dist < farthestCorner and dist > closestCorner):
                    nearby.append(c)
            #now of the corners attempt to find a matching corner
            matchingCorner = None
            for c in nearby:
                angleDiff = self.getAngleDiff(c.angle, currC.angle)
                #3 types can match, 2 normal at 90 degree, 1 of each at 180, or 2 negative where angles sum to a multiple of 90
                if(c.type + currC.type == 2 and abs(angleDiff - 90) <= angleTol): #both 1's, simple end
                    matchingCorner = c
                    anchors.append( ((c.coord + currC.coord)/2).astype(int) ) #midpoint
                    break
                elif(c.type + currC.type == -2 and abs(angleDiff - 90) <= angleTol): #both -1's, T intersection
                    matchingCorner = c
                    anchors.append( self.getIntersection(c,currC).astype(int) )
                    break
                elif(c.type + currC.type == 0 and abs(angleDiff) <= angleTol or abs(angleDiff - 180) <= angleTol): #one of each, L corner
                    matchingCorner = c
                    anchors.append( ((c.coord + currC.coord)/2).astype(int) ) #midpoint
                    break
            if(matchingCorner is not None): #we found a match
                corners.remove(matchingCorner)

        return anchors
    
    #now we need to get anchors from a corner
    #we can do this by associating certain corners together to the same group (e.g. corners of the same edge belong together
    #then the mean position of each group should be the location of the anchor point
    #sampleNum is used for the number of sample to tell if two corners belong to each other (all sample points between should be black)
    #closePercent is the percent distance of the image two corners in the same group can be (e.g. pixels far apart shouldn't belong to the same group)
    def _getAnchorsOld(self, corners, growImg, sampleNum = 5, closePercent=.1): #THIS FUNCTION IS DEPRECATED
        groups = [] #list of groups in which each group is the list of corners belonging to that anchor point
        closeness = np.array(np.shape(growImg))*closePercent
        for corner in corners: #go through each corner and add to existing group or add to new one
            inAnyGroup = False #assume not in any group until proven otherwise
            for group in groups: #check if corner belongs to the group
                inGroup = True #assume in the group until proven otherwise
                for item in group:
                    if( np.any( np.abs(corner - item) > closeness )): #odd way to do this, essentially checks if pixels are too far away
                        inGroup = False
                        continue
                    samples = np.zeros((sampleNum, 2), dtype=int) #find a couple sample points between the corners we're testing, check if they're all black
                    samples[:,0] = np.linspace(corner[0], item[0], sampleNum, dtype=int)
                    samples[:,1] = np.linspace(corner[1], item[1], sampleNum, dtype=int)
                    connected = True #assume the corners are connected until proven otherwise
                    for sample in samples:
                        if(growImg[sample[0], sample[1]] != 1): #this sample fails, since it isn't black
                            connected = False
                            break
                    if(not connected): #corner isn't in the group if it isn't connected to an item in this group
                        inGroup = False
                        break
                if(inGroup): #corner belongs to this group
                    group.append(corner)
                    inAnyGroup = True
                    break #a corner can only belong to one group
            if(not inAnyGroup): #needs a new group
                groups.append([corner])
            
        #groups now contains a list of groups which contains their respective corner coordinates
        anchorPoints = []
        for group in groups:
            anchorPoints.append(np.mean(group, axis=0, dtype=int))
        anchorPoints = np.array(anchorPoints)
        return anchorPoints

#Stroke Mapper code
class StrokeMapper:
    #samples is samples per curve
    def isValidMap(self, oldFit, newFit, img, samples = 30):
        sData = oldFit.data.arial
        orthFit = (sData[0][0][0] == sData[-1][-1][0]) or (sData[0][0][1] == sData[-1][-1][1]) #should be horizontal/vertical 
        #aspect ratio
        oldPerimeter = np.diff(oldFit.x)[0] + np.diff(oldFit.y)[0] * 2
        newPerimeter = np.diff(newFit.x)[0] + np.diff(newFit.y)[0] * 2
        oldCenter = np.array([np.mean(oldFit.x),np.mean(oldFit.y)])
        newCenter = np.array([np.mean(newFit.x),np.mean(newFit.y)])
        centerDist = np.sqrt(sum((oldCenter-newCenter)**2))/len(img)
        #we'll do a bunch of checks to disallow the mapped stoke as an option  
        if(newPerimeter < .25*len(img) or (not orthFit and oldPerimeter*2 < newPerimeter)): #stroke perimeter is too small, or has changed too much
            return False
        if(abs(np.diff(newFit.x)[0]) < len(img)*.01 or abs(np.diff(newFit.y)[0]) < len(img)*.01): #some stroke dimension is too small
            return False
        if(centerDist > .35): #stroke moved to far away
            return False
        if(not orthFit and (newFit.x[0] > newFit.x[1] or newFit.y[0] > newFit.y[1]) ): #stroke is flipped, not allowed, unless flipping doesn't do anything
            return False
        if(not orthFit): #if a stroke is vertical or horizontal we don't care about aspect ratio
            if(newFit.y[0] == newFit.y[1] or newFit.x[0] == newFit.x[1]): #undefined aspect ratio
                return False

        #SData = self.mapStroke(sData, oldFit.x, oldFit.y)
        nSData = self.mapStroke(sData, newFit.x, newFit.y)
        #fit error
        evalPoints = np.linspace(0.0, 1.0, samples) #start, end, resol, parametric t values
        totalSamples = (len(nSData)*samples)
        wrongSamples = 0
        for curve in nSData:
            nodes = np.array(curve).transpose()
            curve = bezier.Curve(nodes, degree=len(curve)-1)
            curvePoints = np.array(curve.evaluate_multi(evalPoints).transpose().tolist()) #[[x,y]...]
            for samplePoint in curvePoints:
                y = int(samplePoint[1])
                x = int(samplePoint[0])
                if(img[y,x] <= 0):
                    wrongSamples += 1
        
        if(wrongSamples / totalSamples > .2): #too many poor sample points
            return False #arbitrary large error

        return True #all checks passed

    def verifyAnchors(self, fit, a1, a2): #some strokes can't be mapped properly, verify points are near an anchor
        fit2 = self.mapStroke(fit.data.arial, fit.x, fit.y)
        beginCoord = fit2[0][0]
        endCoord = fit2[-1][-1]
        beginCoord = np.array([beginCoord[1], beginCoord[0]]) #x,y format
        endCoord = np.array([endCoord[1], endCoord[0]]) #x,y format

        if(np.sum(abs(beginCoord - a1)) > 20 and np.sum(abs(beginCoord - a2)) > 20): #no matching anchor point
            return False
        if(np.sum(abs(endCoord - a1)) > 20 and np.sum(abs(endCoord - a2)) > 20): #no matching anchor point
            return False
        return True

    def mapStroke(self, strokeData, xR, yR):
        strokeData = np.array(strokeData)
        strokeData[:,:,0] = xR[0] + (strokeData[:,:,0]*(xR[1]-xR[0]))
        strokeData[:,:,1] = yR[0] + (strokeData[:,:,1]*(yR[1]-yR[0]))
        return strokeData

    def adjustFit(self, fit, newS, newE, tol=.001): #note newS/newE coord order is [y,x]
        p0 = np.array([fit.x[0], fit.y[0]])
        p1 = np.array([fit.x[1], fit.y[1]])

        fs = np.array(fit.data.arial[0][0])
        fe = np.array(fit.data.arial[-1][-1])

        s = (1-fs)*p0 + fs*p1
        e = (1-fe)*p0 + fe*p1

        d0 = np.zeros((2))
        d1 = np.zeros((2))

        if(abs(fs[0]-fe[0]) >= tol): #check for divide by zero
            d1[0] = ((1-fe[0])*(s[0] - newS[1]) + (1-fs[0])*(newE[1] - e[0]))/(fe[0]-fs[0])
        if(abs(fs[1]-fe[1]) >= tol): #check for divide by zero
            d1[1] = ((1-fe[1])*(s[1] - newS[0]) + (1-fs[1])*(newE[0] - e[1]))/(fe[1]-fs[1])

        if(abs(1-fs[0]) >= tol):
            d0[0] = ((newS[1] - s[0]) - fs[0]*d1[0]) / (1-fs[0])
        elif(abs(1-fe[0]) >= tol):
            d0[0] = ((newE[1] - e[0]) - fe[0]*d1[0]) / (1-fe[0])

        if(abs(1-fs[1]) >= tol):
            d0[1] = ((newS[0] - s[1]) - fs[1]*d1[1]) / (1-fs[1])
        elif(abs(1-fe[1]) >= tol):
            d0[1] = ((newE[0] - e[1]) - fe[1]*d1[1]) / (1-fe[1])
        #d1 = ((1-fe)*(newS - s) + (1-fs)*(newE - e))/(fe-fs)
        #d0 = ((newS - s) - fs*d1) / (1-fs)
        newP0 = p0+d0
        newP1 = p1+d1
        return sf.Fit(fit.name, fit.data, [newP0[0], newP1[0]], [newP0[1], newP1[1]])
        #if(not np.any(np.isclose(fs, fe, atol=.001))): #x or y coordinates don't match
        #    return fit #return as is for now
            #find closest point and only map to that one

    def remapFits(self, character, fitData=None):
        if(fitData is None):
            fitData = fd.loadFits(character)
        img = fi.renderChar(character, size=1000, pad=.1, show=False);
        
        anchConv = AnchorConverter()
        anchors = anchConv.get(img)
        
        fitDataCopy = FitData()
        fitDataCopy.set(fitData.character, [], fitData.size)
        usedPairs = [] #track used anchors so we don't place any points at identical anchors
        
        for origFit in fitData.fits:
            fitFound = None
            
            fitTemp = self.mapStroke(origFit.data.arial, origFit.x, origFit.y)
            beginCoord = fitTemp[0][0]
            endCoord = fitTemp[-1][-1]
            sC = np.array([beginCoord[1], beginCoord[0]]) #y,x format startCoord
            eC = np.array([endCoord[1], endCoord[0]]) #y,x format beginCoord
            
            startDists = [ np.sqrt(sum((sC - a)**2)) for a in anchors]
            endDists = [ np.sqrt(sum((eC - a)**2)) for a in anchors]
            
            zippedStart = zip(startDists, anchors)
            zippedEnd = zip(endDists, anchors)
            
            sortStart = sorted(zippedStart, key = lambda x: x[0]) #sort by distance
            sortEnd = sorted(zippedEnd, key = lambda x: x[0]) #sort by distance
            
            distStart, anchorsStart = zip(*sortStart)
            distEnd, anchorsEnd = zip(*sortEnd)
            
            if(len(anchorsStart) > 5): #only check the n closest anchors
                anchorsStart = anchorsStart[:5]
            if(len(anchorsEnd) > 5): #only check the n closest anchors
                anchorsEnd = anchorsEnd[:5]
                
            #one last zip, now the combination of the two
            anchorPairs = []
            pairDists = []
            for i in range(len(anchorsStart)):
                for j in range(len(anchorsEnd)):
                    pairDists.append(distStart[i] + distEnd[j])
                    anchorPairs.append([anchorsStart[i], anchorsEnd[j]])
                    
            zippedPair = zip(pairDists, anchorPairs)
            sortPair = sorted(zippedPair, key = lambda x: x[0]) #sort by distance
            distPair, anchorPairs = zip(*sortPair)
            
            for aPair in anchorPairs:
                a1 = aPair[0]
                a2 = aPair[1]
                if(np.all(a1 == a2)):
                    continue
                
                identicalAnchors = False
                for pair in usedPairs:
                    if( ( np.all(a1 == pair[0]) and np.all(a2 == pair[1])  ) or (np.all(a1 == pair[1]) and np.all(a2 == pair[0]))):
                        identicalAnchors = True
                        break
                if(identicalAnchors): #we've used this pair before
                    continue
                
                newFit = self.adjustFit(origFit, a1, a2)
                if(not self.verifyAnchors(newFit,a1,a2)): #map didn't complete
                    continue
                if(not self.isValidMap(origFit, newFit, img)):
                    continue
                    
                newFit = self.adjustOrthoFit(newFit, width = len(img)/10)
                usedPairs.append([a1, a2])
                fitFound = newFit
                break
                
            if(fitFound is not None):
                newFitType = sf.Fit(fitFound.name, fitFound.data, fitFound.x, fitFound.y)
            else: #use original fit
                newFitType = sf.Fit(origFit.name, origFit.data, origFit.x, origFit.y)
            fitDataCopy.fits.append(newFitType)
        return fitDataCopy
    
    def adjustOrthoFit(self, fit, width=50):
        sData = fit.data.arial
        vert = (sData[0][0][0] == sData[-1][-1][0])
        horiz = (sData[0][0][1] == sData[-1][-1][1])
        width = width//2 #how far we go on each side
        
        if(horiz):
            center = (fit.y[0] + fit.y[1]) // 2
            fit.y = [center - width, center + width]
        if(vert):
            center = (fit.x[0] + fit.x[1]) // 2
            fit.x = [center - width, center + width]
        return fit
    
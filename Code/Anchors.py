import numpy as np

#handles converting an img to the anchor points
class AnchorConverter:
    def __init__(self):
        
        #for getDiff
        self.DiffBorderDist=5 #borderDiff is how far to check away from each pixel
        
        #for get Corners
        self.diffThresh=.25 #ratio diff between white/black needed to be considered for a corner
        self.borderDiff=5 #how big a distance we look for a local minimum (smaller may mean more corners detected)
        self.cornerPrint=False #prints info about the lowest passed pixel threshhold
        
        #for growImg
        self.growDist = 3 #for the grown image how many pixels to grow the image
        
        #for getAnchors
        self.connectedSampleNum = 5 #how many samples to take to check if corners are connected
        self.closePercent = .1 #up to this % of the image span may allow corners to be considered to be in the same group
        
    #get and return the anchor points
    def get(self, img, getCorners=False):
        #get diff img, difference between nearby white/black neighbors
        diffImg = self._getDiff(img, borderDiff=self.DiffBorderDist)
        #get a list of corner coordinates, which we'll use to determine the anchors
        corners = self._getCorners(diffImg, diffThresh=self.diffThresh, borderDiff=self.borderDiff, printInfo=self.cornerPrint)
        #get a grown version of the image, used for checking if corners belong to the same anchor point
        growImg = self._getGrown(img, growDist=self.growDist)
        anchors = self._getAnchors(corners, growImg, sampleNum=self.connectedSampleNum, closePercent=self.closePercent)
        
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
        diffImg = np.abs(diffImg)

        #now determine which pixels are on the border (black but touches a white pixel, corner doesn't count)
        borderImg = np.zeros(np.shape(img)) #zero is not a border
        borderImg[1:-1, 1:-1][(img[1:-1, 1:-1] == 1) & ( img[:-2, 1:-1] == 0)] = 1 #up direction
        borderImg[1:-1, 1:-1][(img[1:-1, 1:-1] == 1) & (img[2:, 1:-1] == 0)] = 1 #down direction
        borderImg[1:-1, 1:-1][(img[1:-1, 1:-1] == 1) & (img[1:-1, :-2] == 0)] = 1 #left direction
        borderImg[1:-1, 1:-1][(img[1:-1, 1:-1] == 1) & (img[1:-1, 2:] == 0)] = 1 #right direction

        diffImg[borderImg != 1] = 0 #reset any pixels that aren't a border
        diffImg /= (borderDiff*2)**2 - 1 #ratio of total pixels counted
        return diffImg

    #from the diff image find the pixels which are the "corners" of the character
    #diffThresh is the relative difference between black and white for a pixel to be considered as a corner (recommended: 25%)
    #borderDiff is how far to check away from each pixel for a better ratio
    def _getCorners(self, diffImg, diffThresh = .25, borderDiff=5, printInfo=True):
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

        print("Lowest ratio corner:", min(diffImg[maxImg == 1])) #lowest diff that passed the test
        print("Highest ratio corner:", max(diffImg[maxImg == 1])) #highest diff that passed the test

        corners = np.array(np.column_stack(np.where(maxImg == 1))) #corners like: [ [corner1Y, corner1X], [corner2Y, corner2X], [corner3Y, corner3X] ... ]
        return corners

    
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
    
    #now we need to get anchors from a corner
    #we can do this by associating certain corners together to the same group (e.g. corners of the same edge belong together
    #then the mean position of each group should be the location of the anchor point
    #sampleNum is used for the number of sample to tell if two corners belong to each other (all sample points between should be black)
    #closePercent is the percent distance of the image two corners in the same group can be (e.g. pixels far apart shouldn't belong to the same group)
    def _getAnchors(self, corners, growImg, sampleNum = 5, closePercent=.1):
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

    
    
    
    
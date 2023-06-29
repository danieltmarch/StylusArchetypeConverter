import numpy as np
import matplotlib.pyplot as plt
import bezier
import re #regex
import pickle #for saving objects
import os #for searching for files in a directory

def plotStroke(stroke):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.invert_yaxis()
    
    evalPoints = np.linspace(0.0, 1.0, 100) #start, end, resol, parametric t values
    for st in stroke: #control points, 
        #graph actual curve (bezier, use package)
        nodes = np.array(st).transpose()
        curve = bezier.Curve(nodes, degree=len(st)-1)
        curvePoints = np.array( curve.evaluate_multi(evalPoints).transpose().tolist()) #[[x,y]...]
        ax.plot(curvePoints[:,0], curvePoints[:,1], color="black")
        
    return fig, ax, curvePoints


#strokeDef contains only the point data, is a list of segments, segments is list of coordinates, coordinate is list of [x,y] lists
#type is the def of both formats and a name
#symbol is the ascii character that looks closes, e.g. ㇀
class StrokeType:
    def set(self, name, symbol, arialDef, hanziDef): #name should be suitable to be a file name, e.g. no slashes or question marks
        self.name = name
        self.symbol = symbol
        
        #so we can use numpy, we'll force the stroke defs to be "square" (all sublists of equal lenghths), just duplicate end points
        maxArialLen = max(len(subList) for subList in arialDef)
        for i in range(len(arialDef)):
            while(len(arialDef[i]) != maxArialLen):
                arialDef[i].append(arialDef[i][-1])
        maxHanziLen = max(len(subList) for subList in hanziDef)
        for i in range(len(hanziDef)):
            while(len(hanziDef[i]) != maxHanziLen):
                hanziDef[i].append(hanziDef[i][-1])
        self.arial = np.array(arialDef)
        self.hanzi = np.array(hanziDef)

    def copy(self): #deep copy all the data
        newStroke = StrokeType()
        newStroke.name = self.name
        newStroke.symbol = self.symbol
        newStroke.arial = []
        for seg in self.arial:
            newSeg = []
            for point in seg:
                newSeg.append([point[0], point[1]])
            newStroke.arial.append(newSeg)
        newStroke.hanzi = []
        for seg in self.hanzi:
            newSeg = []
            for point in seg:
                newSeg.append([point[0], point[1]])
            newStroke.hanzi.append(newSeg)
        return newStroke

def getFileNameFromStroke(stroke):
    return re.sub("[^a-zA-Z\d\s]", "", stroke.name) #allow only characters, digits, and spaces
def loadStroke(fileName, extension=".pickle"):
    try:
        with open(f"../Data/Strokes/{fileName}{extension}", 'rb') as file:
            return pickle.load(file)
    except IOError:
        return None
def saveStroke(stroke):
    fileName = getFileNameFromStroke(stroke)
    with open(f"../Data/Strokes/{fileName}.pickle", 'wb') as file:
        pickle.dump(stroke, file)
        
def loadAllStrokes():
    allStrokes = []
    for fileName in os.listdir("../Data/Strokes"):
        if(".pickle" in fileName): #otherwise skip file
            allStrokes.append(loadStroke(fileName, extension=""))
    return allStrokes

def loadStrokeDict(): #load strokes as a dictionary, name is the key, value is the stroke type
    strokeDict = {}
    for fileName in os.listdir("../Data/Strokes"):
        if(".pickle" in fileName): #otherwise skip file
            stroke = loadStroke(fileName, extension="")
            strokeDict[stroke.name] = stroke
    return strokeDict
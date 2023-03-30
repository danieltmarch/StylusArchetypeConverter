import numpy as np
import matplotlib.pyplot as plt
import bezier
import re #regex
import pickle

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
        self.arial = arialDef
        self.hanzi = hanziDef

def getFileNameFromStroke(stroke):
    return re.sub("[^a-zA-Z\d\s]", "", stroke.name) #allow only characters, digits, and spaces
def loadStroke(fileName, extension=".pickle"):
    with open(f"../Data/Strokes/{fileName}{extension}", 'rb') as file:
        return pickle.load(file)
def saveStroke(stroke):
    fileName = getFileNameFromStroke(stroke)
    with open(f"../Data/Strokes/{fileName}.pickle", 'wb') as file:
        pickle.dump(stroke, file)
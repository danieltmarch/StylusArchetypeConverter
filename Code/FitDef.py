#fit data handles saving a characters fits and relevant info
#this is the main format the app uses to draw and adjust
import numpy as np
import matplotlib.pyplot as plt
from StrokeFit import Fit
import StrokeFit as sf
import pickle #for saving

class FitData:
    #character, the chinese character for this particular fit
    #fits, a list of Fit types with bounding boxes for the arial unicode image
    #size: [width, height] for the arial unicode image
    def set(self, character, fits, size): #same thing as init
        self.character = character
        self.fits = fits
        self.size = np.array(size)
        
#function to save/load a particular fit
def loadFits(character):
    try:
        with open(f"../Data/Fits/{character}.pickle", 'rb') as file:
            return pickle.load(file) #FitData
    except IOError:
        return None
def saveFits(fitData):
    with open(f"../Data/Fits/{fitData.character}.pickle", 'wb') as file:
        pickle.dump(fitData, file)
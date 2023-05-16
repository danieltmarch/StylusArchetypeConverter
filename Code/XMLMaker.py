from xml.etree import ElementTree as ET
import FitDef as fd

def save(char):
    charData = fd.loadFits(char)
    #an xml file needs to be created and then saved in ../Data/XML/, see XML Test for an example of starting out.
    print(f"{char} xml save function called!")
#runs the basic tkinter app

import bezier
from tkinter import *
import numpy as np
from PIL import ImageGrab

currChar = [] #list of strokes
currStroke = [] #list of control points

#EVENT FUNCTIONS
def clickEventPoint(event): #click event
    global currChar, currStroke #force global vars
    
    #print("clicked at", event.x, event.y)
    tempChar = currChar.copy()
    currStroke.append([event.x, event.y])
    tempChar.append(currStroke)
    graphChar(tempChar)
    #print(tempChar)
def clickEventStroke(event): #click event
    global currChar, currStroke #force global vars
    
    currChar.append(currStroke)
    currStroke = [] #new stroke
    graphChar(currChar)

def clickEventClear(event): #click event
    global currChar, currStroke, drawChar #force function to use global vars
    if(event != 0 and event.char=='c'):
        currChar = [] #reset lists
        currStroke = []
        graphChar(currChar) #just redraw with the now empty currChar data
    
def graphChar(strokeList): #draw the char
    global currChar, currStroke, drawChar, canvas #force function to use global vars
    
    pr = 3 #point radius size
    lineWidth = 35
    evalPoints = np.linspace(0.0, 1.0, 50) #start, end, resol, parametric t values
    canvas.delete("all") #clear the canvas of all objects
    canvas.create_text(400, 400, text=drawChar, font=('Microsoft YaHei','500'), fill='#555555')
    for stroke in strokeList:
        #graph control points
        for point in stroke: #control points, 
            canvas.create_oval(point[0]-pr, point[1]-pr, point[0]+pr, point[1]+pr, fill='#00AA00')
        if(len(stroke)>1):
            canvas.create_line(stroke, fill ='#00AA00', width=1)
        #graph actual curve (bezier, use package)
        nodes = np.array(stroke).transpose()
        curve = bezier.Curve(nodes, degree=len(stroke)-1)
        curvePoints = curve.evaluate_multi(evalPoints).transpose().tolist() #[x_value list, y_value list]
        canvas.create_line(curvePoints, fill='#000000', width=lineWidth)
        
def onTextChange(event): #user entered a char
    global drawChar, charEntry
    drawChar = charEntry.get()
    clickEventClear(0) #default arg of 0 as the event arg isn't used
    
def onTextClear(event): #user is going to enter a char, clear the current char (1 max char in entry box)
    global entryText
    entryText.set('')
#END - EVENT FUNCTIONS
    
#from: https://www.pythonfixing.com/2022/06/fixed-how-to-save-tkinter-canvas-as.html
def saveCanvasAsImage(): #literally screen captures 
    global canvas, root
    x=root.winfo_rootx()+canvas.winfo_x()
    y=root.winfo_rooty()+canvas.winfo_y()
    x1=x+canvas.winfo_width()
    y1=y+canvas.winfo_height()
    #print(x,y,x1,y1)
    ImageGrab.grab().crop((x,y,x1,y1)).save("Images/tempChar.png")

#canvas for drawing characters and strokes (defined here so it can be used as global var)
canvas = 0
drawChar = '㇀'
root = 0
charEntry = 0
entryText = 0

#actually run the app
def run():
    global canvas, root, currChar, currStroke, charEntry, entryText
    
    #define global canvas and stuff
    canvasSize = 800

    # Create object and size
    root = Tk()
    root.geometry( "800x850" )

    #add label for draw char, label, text
    charLabel = Label(root, text = "Char: ")
    charLabel.pack()

    #handle user typing in a char
    entryText = StringVar()
    charEntry = Entry(root, width = 10, justify="center", textvariable=entryText)
    entryText.set(drawChar) #default char
    charEntry.bind("<KeyPress>", onTextClear) #user is about to finish typing a char
    charEntry.bind("<KeyRelease>", onTextChange) #user entered a char
    charEntry.pack()

    #button for processing the button
    processBtn = Button(root, text ="Gen. Strokes", command = saveCanvasAsImage)
    processBtn.pack()
    
    #setup canvas
    canvas = Canvas(root, bg="white", height=canvasSize, width=canvasSize) #our drawing board
    canvas.create_text(400, 400, text=drawChar, font=('Microsoft YaHei','500'), fill='#555555')
    clickEventClear(0) #default arg of 0 as the event arg isn't used
    canvas.pack()
    
    #reset the lists before we re-run the tkinter app
    currChar = []
    currStroke = []

    canvas.bind("<Button-1>", clickEventPoint) #setup the left click event, add a control point
    root.bind("<Key>", clickEventClear) #setup the left click event, clear the canvas
    canvas.bind("<Double-Button-1>", clickEventStroke) #setup the right click event, finalize stroke
    
    root.mainloop() #run the app
    
    return currChar
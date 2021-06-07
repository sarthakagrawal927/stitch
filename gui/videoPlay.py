from tkinter import *
from tkvideo import tkvideo
from tkinter import filedialog
from video_stitcher4 import *

root = Tk()
root.attributes('-fullscreen', True) # make main window full-screen

# canvas = Canvas(root, bg="blue", height=1000, width=1000)
my_label = Label(root)

def playVideo(path):
    player = tkvideo(path, my_label, loop = 1, size = (1280,720))
    player.play()

def browserFunction():
    filename = filedialog.askopenfilename()
    pathlabel1.config(text=filename)

def browserFunction2():
    filename = filedialog.askopenfilename()
    pathlabel2.config(text=filename)

def getVideo():
    newFile = main(pathlabel1['text'],pathlabel2['text'])
    print(newFile)
    playVideo(newFile)

browsebutton = Button(root, text="Browse 1st", command=browserFunction)
browsebutton.pack()

pathlabel1= Label(root)
pathlabel1.pack()

browsebutton2 = Button(root, text="Browse 2nd", command=browserFunction2)
browsebutton2.pack()

pathlabel2 = Label(root)
pathlabel2.pack()

getOutput = Button(root, text="Output", command=getVideo)
getOutput.pack()

Button(root, text="Quit", command=root.destroy).pack()

my_label.pack()
# canvas.pack()
root.mainloop()

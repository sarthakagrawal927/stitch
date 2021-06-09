from tkinter import *
from tkvideo import tkvideo
from tkinter import filedialog
from video_stitcher4 import *

root = Tk()
root.attributes('-fullscreen', True) # make main window full-screen

# canvas = Canvas(root, bg="blue", height=1000, width=1000)
solution_video = Label(root)
video1 = Label(root)
video2= Label(root)

heading = Label(root, text="Video Stitching", bg = "black", fg="white")
heading.grid(row = 0, column=1)

def playVideo(path , labelName, labelSize):
    player = tkvideo(path, labelName, loop = 1, size = labelSize)
    player.play()

def browserFunction():
    filename = filedialog.askopenfilename()
    pathlabel1.config(text=filename)

def browserFunction2():
    filename = filedialog.askopenfilename()
    pathlabel2.config(text=filename)

def getVideo():
    newFile = main(pathlabel1['text'],pathlabel2['text'])
    playVideo(newFile, solution_video, (900,600))
    playVideo(pathlabel1['text'], video1, (480,320))
    playVideo(pathlabel2['text'], video2, (480,320))


browsebutton = Button(root, text="Browse 1st", command=browserFunction)
browsebutton.grid(row= 1,column=0)

pathlabel1 = Label(root)
pathlabel1.grid(row=2,column=0)

browsebutton2 = Button(root, text="Browse 2nd", command=browserFunction2)
browsebutton2.grid(row=1,column=2)

pathlabel2 = Label(root)
pathlabel2.grid(row=2,column=2)

getOutput = Button(root, text="Output", command=getVideo)
getOutput.grid(row=3,column=1)

Button(root, text="Quit", command=root.destroy).grid(row =4,column=1)

solution_video.grid(row = 5,column=1)
video1.grid(row = 4,column=0)
video2.grid(row = 4,column=2)

# canvas.pack()
root.mainloop()

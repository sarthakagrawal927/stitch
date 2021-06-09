from tkinter import *
from tkvideo import tkvideo
from tkinter import filedialog,messagebox
from video_stitcher4 import *

root = Tk()
root.attributes('-fullscreen', True) # make main window full-screen

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
    messagebox.showinfo("Output","Hi Output is ready, press ok to play")
    playVideo(newFile, solution_video, (900,600))
    playVideo(pathlabel1['text'], video1, (480,320))
    playVideo(pathlabel2['text'], video2, (480,320))

solution_video = Label(root)
video1 = Label(root)
video2= Label(root)

heading = Label(root, text="Video Stitching", bg = "black", fg="white")

browsebutton = Button(root, text="Browse 1st", command=browserFunction)
pathlabel1 = Label(root)

browsebutton2 = Button(root, text="Browse 2nd", command=browserFunction2)
pathlabel2 = Label(root)

getOutput = Button(root, text="Stitch", command=getVideo)

# showOutput = Button(root, text="Play", command=getVideo)
# showOutput.grid(row=3,column=2)

heading.grid(row = 0, column=2)

browsebutton.grid(row= 1,column=0)
pathlabel1.grid(row=1,column=1)
browsebutton2.grid(row=1,column=3)
pathlabel2.grid(row=1,column=4)

getOutput.grid(row=2,column=2)
video1.grid(row = 2,column=0,columnspan=2, rowspan=2)
video2.grid(row = 2,column=3,columnspan=2, rowspan=2)
Button(root, text="Quit", command=root.destroy).grid(row =3,column=2)

solution_video.grid(row = 5,column=0, columnspan= 5)

root.mainloop()

import sys,json
from video_stitcher5 import *

data = sys.stdin.readlines()
paths = data[0].split(',')
path1 = paths[0]
path2 = paths[1][:-1]
print(path1)
print(path2)
outPath = main(path1,path2)
print(outPath)
sys.stdout.flush()
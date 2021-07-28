# Sarthak's cut

import cv2
import numpy as np
import imutils
import tqdm
import os

class VS:
    def __init__(self, left_video_in_path, right_video_in_path, video_out_path, display = False):
        self.left_video_in_path = left_video_in_path
        self.right_video_in_path = right_video_in_path
        self.video_out_path = video_out_path
        self.display = display

        self.saved_homography_matrix = None

    def stitch(self,images, ratio, reproj_thresh):
        image1, image2 = images

        

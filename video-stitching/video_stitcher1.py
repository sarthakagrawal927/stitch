# using complete stitcher class

import cv2
import numpy as np
import imutils
import tqdm
import os


class VideoStitcher:
    def __init__(self, left_video_in_path, right_video_in_path, video_out_path, video_out_width=800, display=False):
        # Initialize arguments
        self.left_video_in_path = left_video_in_path
        self.right_video_in_path = right_video_in_path
        self.video_out_path = video_out_path
        self.video_out_width = video_out_width
        self.display = display

        # Initialize the saved homography matrix
        self.saved_homo_matrix = None

    def run(self):
        # Set up video capture
        left_video = cv2.VideoCapture(self.left_video_in_path)
        right_video = cv2.VideoCapture(self.right_video_in_path)
        print('[INFO]: {} and {} loaded'.format(self.left_video_in_path.split('/')[-1],
                                                self.right_video_in_path.split('/')[-1]))
        print('[INFO]: Video stitching starting....')

        # Get information about the videos
        n_frames = min(int(left_video.get(cv2.CAP_PROP_FRAME_COUNT)),
                       int(right_video.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = int(left_video.get(cv2.CAP_PROP_FPS))
        frames = []

        for _ in tqdm.tqdm(np.arange(n_frames)):
            # Grab the frames from their respective video streams
            ok, left = left_video.read()
            _, right = right_video.read()

            # resizing
            # r_left = 1200 / left.shape[1]
            # dim_left = (1200, int(left.shape[0] * r_left))

            # r_right = 1200 / right.shape[1]
            # dim_right = (1200, int(right.shape[0] * r_right))

            # left = cv2.resize(left, dim_left)
            # right = cv2.resize(right, dim_right)

            # print(left.shape[0], left.shape[1])
            # print(right.shape[0], right.shape[1])
            left = cv2.rotate(left, cv2.ROTATE_90_CLOCKWISE)

            right = cv2.rotate(right, cv2.ROTATE_90_CLOCKWISE)

            cv2.imshow("image1",left)
            cv2.imshow("image2",right)

            images = [left, right]

            if ok:
                # Stitch the frames together to form the panorama
                stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
                (status, stitched) = stitcher.stitch(images)

                # No homography could not be computed
                if status != 0:
                    print("[INFO]: Homography could not be computed!")
                    break

                # Add frame to video
                stitched = imutils.resize(
                    stitched, width=self.video_out_width)

                frames.append(stitched)
            print(frames)
            if self.display:
                # Show the output images
                cv2.imshow("Result", stitched)

            # If the 'q' key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

        # Save video
        print('[INFO]: Saving {} in {}'.format(self.video_out_path.split('/')[-1],
                                               os.path.dirname(self.video_out_path)))

        frame1 = frames[0]
        height, width, layers = frame1.shape

        # print(frames.__len__())
        # print(fps)

        clip = cv2.VideoWriter(self.video_out_path, 0,
                               fps/frames.__len__(), (width, height))

        for frame in frames:
            clip.write(frame)

        clip.release()


# Example call to 'VideoStitcher'
stitcher = VideoStitcher(left_video_in_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/3/2.mp4',
                         right_video_in_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/3/4.mp4',
                         video_out_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/3_1.mp4')

stitcher.run()

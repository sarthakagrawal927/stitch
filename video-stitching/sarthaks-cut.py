# Sarthak's cut

import cv2
import numpy as np
import imutils
import tqdm
import os

reproj_thresh = 100.0
lowes_ratio = 1.4
minMatches = 5
currentFrameShow = True
needClockWiseRotation = True
showFeatureMatching = True

class VideoStitcher:
    def __init__(self, left_video_in_path, right_video_in_path, video_out_path):
        self.left_video_in_path = left_video_in_path
        self.right_video_in_path = right_video_in_path
        self.video_out_path = video_out_path
        self.saved_homography_matrix = None

    def stitch(self,images):
        (image_left, image_right) = images

        if needClockWiseRotation:
            image_left = cv2.rotate(image_left, cv2.ROTATE_90_CLOCKWISE)
            image_right = cv2.rotate(image_right, cv2.ROTATE_90_CLOCKWISE)

        if currentFrameShow:
            cv2.imshow("left", image_left)
            cv2.imshow("right", image_right)

        (height_left, width_left, channels_left) = image_left.shape
        (height_right, width_right, channels_right) = image_right.shape
        output_shape = width_left + width_right, height_left

        if self.saved_homography_matrix is None:

            (keypoints_left, features_left) = self.detect_and_extract(image_left)
            (keypoints_right, features_right) = self.detect_and_extract(image_right)

            matched_keypoints = self.match_keypoints(keypoints_left, keypoints_right, features_left, features_right)

            if matched_keypoints is None:
                return None
            print("here")
            if showFeatureMatching:
                visual=self.draw_matches(image_left, image_right, keypoints_left, keypoints_right, matched_keypoints[0], matched_keypoints[2])
                cv2.imwrite("matching.png",imutils.resize(visual, width=output_shape[0]))

            self.saved_homography_matrix = matched_keypoints[1]

        # warpPerspective will place the first image on a plane surface based on  the matching keypoints with the
        # second image as computed by the homography matrix
        result_image = cv2.warpPerspective(image_left, self.saved_homography_matrix, output_shape)

        # concatinating the image right to the result
        result_image[0:height_right, 0:width_right] = image_right

        return result_image

    @staticmethod
    def detect_and_extract(image):
        # Extracting features from images using SIFT algorithm
        descriptor = cv2.xfeatures2d.SIFT_create()

        # mask is None here as we do not to mask anything, we use mask values when we need to find a particular feature
        (keypoints, features) = descriptor.detectAndCompute(image,None)

        # Keypoint objects to numpy arrays
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        return (keypoints, features)

    @staticmethod
    def match_keypoints(keypoints_a, keypoints_b, features_a, features_b):
        # https://docs.opencv.org/4.5.2/dc/dc3/tutorial_py_matcher.html
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        # matcher = cv2.BFMatcher()

        raw_matches = matcher.knnMatch(features_a,features_b, k = 2)
        matches = []

        for raw_match in raw_matches:
            # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(raw_match) == 2 and raw_match[0].distance < raw_match[1].distance * lowes_ratio:
                matches.append((raw_match[0].trainIdx , raw_match[0].queryIdx))

        if len(matches) > minMatches:
            points_a = np.float32([keypoints_a[i] for (_, i) in matches])
            points_b = np.float32([keypoints_b[i] for (i, _) in matches])

            homography_matrix, status = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_thresh)

            return (matches, homography_matrix, status)

        return None

    @staticmethod
    def draw_matches(image_a, image_b, keypoints_a, keypoints_b, matches, status):
        # Initialize the output visualization image
        (height_a, width_a) = image_a.shape[:2]
        (height_b, width_b) = image_b.shape[:2]
        visualisation = np.zeros(
            (max(height_a, height_b), width_a + width_b, 3), dtype="uint8")
        visualisation[0:height_a, 0:width_a] = image_a
        visualisation[0:height_b, width_a:] = image_b

        for ((train_index, query_index), s) in zip(matches, status):
            # Only process the match if the keypoint was successfully matched
            if s == 1:
                # Draw the match
                point_a = (int(keypoints_a[query_index][0]), int(
                    keypoints_a[query_index][1]))
                point_b = (
                    int(keypoints_b[train_index][0]) + width_a, int(keypoints_b[train_index][1]))
                cv2.line(visualisation, point_a, point_b, (0, 255, 0), 1)

        # return the visualization
        return visualisation

    def run(self):
        left_video = cv2.VideoCapture(self.left_video_in_path)
        right_video = cv2.VideoCapture(self.right_video_in_path)

        frames_video1 = left_video.get(cv2.CAP_PROP_FRAME_COUNT)
        frames_video2 = right_video.get(cv2.CAP_PROP_FRAME_COUNT)
        print("frames ",frames_video1,frames_video2)
        n_frames = int(min(frames_video1, frames_video2))

        fps_video1 = left_video.get(cv2.CAP_PROP_FPS)
        fps_video2 = right_video.get(cv2.CAP_PROP_FPS)
        print("fps ",fps_video1,fps_video2)
        n_fps = min(fps_video1, fps_video2)

        frames = []

        for _ in tqdm.tqdm(np.arange(n_frames)):

            left_ok, left = left_video.read()
            right_ok, right = right_video.read()

            if left_ok and right_ok:
                stitched_frame = self.stitch([left,right])

                if stitched_frame is None:
                    print("Homography -> not computed : not enough matching points")
                    break

                if currentFrameShow:
                    cv2.imshow("Stitched", stitched_frame)

                # stitched_frame = imutils.resize(stitched_frame, width=self.video_out_width)
                frames.append(stitched_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()

stitcher = VideoStitcher(left_video_in_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/poster/1.mp4',
                         right_video_in_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/poster/2.mp4',
                         video_out_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/poster/1234.mp4')

stitcher.run()
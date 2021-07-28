# the current one

import cv2
import numpy as np
import imutils
import tqdm
import os

class VideoStitcher:
    def __init__(self, left_video_in_path, right_video_in_path, video_out_path, video_out_width=1200, display=True):
        # Initialize arguments
        self.left_video_in_path = left_video_in_path
        self.right_video_in_path = right_video_in_path
        self.video_out_path = video_out_path
        self.video_out_width = video_out_width
        self.display = display

        # Initialize the saved homography matrix
        self.saved_homo_matrix = None

    def stitch(self, images, ratio=0.75, reproj_thresh=100.0):
        # Unpack the images
        (image_a, image_b) = images
        image_a = cv2.rotate(image_a, cv2.ROTATE_90_CLOCKWISE)
        image_b = cv2.rotate(image_b, cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow("image1",image_a)
        cv2.imshow("image2",image_b)

        # image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
        # image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
        # If the saved homography matrix is None, then we need to apply keypoint matching to construct it
        if self.saved_homo_matrix is None:
            # Detect keypoints and extract
            (keypoints_a, features_a) = self.detect_and_extract(image_a)
            (keypoints_b, features_b) = self.detect_and_extract(image_b)

            # Match features between the two images
            matched_keypoints = self.match_keypoints(
                keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

            # If the match is None, then there aren't enough matched keypoints to create a panorama
            if matched_keypoints is None:
                return None

            # Save the homography matrix
            self.saved_homo_matrix = matched_keypoints[1]
            visual=self.draw_matches(image_a, image_b, keypoints_a, keypoints_b, matched_keypoints[0], matched_keypoints[2])
            cv2.imwrite("matching.png",imutils.resize(visual, width=1000))
        # Apply a perspective transform to stitch the images together using the saved homography matrix
        output_shape = (image_b.shape[1] + image_a.shape[1], image_b.shape[0])

        result = cv2.warpPerspective(
            image_b, self.saved_homo_matrix, output_shape)

        result[0:image_a.shape[0], 0:image_a.shape[1]] = image_a
        # Return the stitched image
        return result

    @staticmethod
    def detect_and_extract(image):

        # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
        descriptor = cv2.xfeatures2d.SIFT_create()
        (keypoints, features) = descriptor.detectAndCompute(image, None)

        # #SURF
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # surf = cv2.SURF_create()
        # keypoints, features = surf.detectAndCompute(gray_img, None)


        # #FAST - only keypoint detector
        # fast = cv2.FastFeatureDetector_create(threshold=25)
        # # # find and draw the keypoints
        # kp = fast.detect(image,None)
        # # BRISK - extract features
        # br = cv2.BRISK_create();
        # keypoints, features = br.compute(image,  kp)

        #ORB
        # gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # orb = cv2.ORB_create(nfeatures=2000)
        # keypoints, features = orb.detectAndCompute(gray_img, None)

        # Convert the keypoints from KeyPoint objects to numpy arrays
        keypoints = np.float32([keypoint.pt for keypoint in keypoints])

        # Return a tuple of keypoints and features
        return (keypoints, features)

    @staticmethod
    def match_keypoints(keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh):
        # Compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, k=2)
        matches = []

        for raw_match in raw_matches:
            # Ensure the distance is within a certain ratio of each other (i.e. Lowe's ratio test)
            if len(raw_match) == 2 and raw_match[0].distance < raw_match[1].distance * ratio:
                matches.append((raw_match[0].trainIdx, raw_match[0].queryIdx))

        # Computing a homography requires at least 4 matches
        if len(matches) > 10:
            # Construct the two sets of points
            points_a = np.float32([keypoints_a[i] for (_, i) in matches])
            points_b = np.float32([keypoints_b[i] for (i, _) in matches])

            # Compute the homography between the two sets of points
            (homography_matrix, status) = cv2.findHomography(
                points_a, points_b, cv2.RANSAC, reproj_thresh)

            # Return the matches, homography matrix and status of each matched point
            return (matches, homography_matrix, status)

        # No homography could be computed
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
        i=0
        clip = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                    30, (500, 500))
        for _ in tqdm.tqdm(np.arange(n_frames)):
            # Grab the frames from their respective video streams
            ok, left = left_video.read()
            _, right = right_video.read()

            if ok and _ :
                # Stitch the frames together to form the panorama
                stitched_frame = self.stitch([left, right])

                # No homography could not be computed
                if stitched_frame is None:
                    print("[INFO]: Homography could not be computed!")
                    break

                # Add frame to video
                stitched_frame = imutils.resize(
                    stitched_frame, width=self.video_out_width)

                #frames.append(stitched_frame)
                if i == 0 :
                    height, width, layers = stitched_frame.shape
                    clip = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*'avc1'),
                                    30, (width, height))
                    i=1

                clip.write(stitched_frame)

                if self.display:
                    cv2.imshow("Result", stitched_frame)

            # If the 'q' key was pressed, break from the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

        # Save video
        print('[INFO]: Saving {} in {}'.format(self.video_out_path.split('/')[-1],
                                               os.path.dirname(self.video_out_path)))

        # height, width, layers = frames[0].shape

        # clip = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*'mp4v'),
        #                        30, (width, height))

        # for frame in frames:
        #     clip.write(frame)

        clip.release()
        print('[INFO]: {} saved'.format(self.video_out_path.split('/')[-1]))


stitcher = VideoStitcher(left_video_in_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/2/2.mp4',
                         right_video_in_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/2/3.mp4',
                         video_out_path='/Users/sarthakagrawal/Desktop/stitch/SamsungInput/test/2_1.mp4')

stitcher.run()
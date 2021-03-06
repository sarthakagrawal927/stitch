# Github Code with color coding fixed

import cv2
import numpy as np
import imutils
import tqdm
import os

minMatches = 50
currentFrameShow = False
needClockWiseRotation = False
showFeatureMatching = False

def trim(frame):
    #crop top
    if not np.sum(frame[0]):
        return trim(frame[1:])
    #crop bottom
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    #crop left
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    #crop right
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame

class VideoStitcher:
    def __init__(self, left_video_in_path, right_video_in_path, video_out_path, video_out_width=800):
        # Initialize arguments
        self.left_video_in_path = left_video_in_path
        self.right_video_in_path = right_video_in_path
        self.video_out_path = video_out_path
        self.video_out_width = video_out_width

        # Initialize the saved homography matrix
        self.saved_homo_matrix = None

    def stitch(self, images, ratio=0.75, reproj_thresh=100.0):
        # Unpack the images
        (image_b, image_a) = images

        if needClockWiseRotation is True:
            image_a = cv2.rotate(image_a, cv2.ROTATE_90_CLOCKWISE)
            image_b = cv2.rotate(image_b, cv2.ROTATE_90_CLOCKWISE)

        cv2.imshow("image1",image_a)
        cv2.imshow("image2",image_b)
        output_shape = ((int)((image_a.shape[1] + image_b.shape[1])/1.0), max(image_a.shape[0], image_b.shape[0]))


        # image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
        # image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
        # If the saved homography matrix is None, then we need to apply keypoint matching to construct it

        if self.saved_homo_matrix is None:
        # if None is None:

            # Detect keypoints and extract
            (keypoints_a, features_a) = self.detect_and_extract(image_a)
            (keypoints_b, features_b) = self.detect_and_extract(image_b)

            # Match features between the two images
            matched_keypoints = self.match_keypoints(
                keypoints_a, keypoints_b, features_a, features_b, ratio, reproj_thresh)

            # If the match is None, then there aren't enough matched keypoints to create a panorama
            if matched_keypoints is None:
                return None

            if showFeatureMatching is True:
                visual=self.draw_matches(image_b, image_a, keypoints_a, keypoints_b, matched_keypoints[0], matched_keypoints[2])
                cv2.imwrite("matching.png",imutils.resize(visual, output_shape[1]))
            # Save the homography matrix
            self.saved_homo_matrix = matched_keypoints[1]
            print(image_a.shape, image_b.shape, output_shape)


        # Apply a perspective transform to stitch the images together using the saved homography matrix
        # print(image_a.shape)
        # print(image_b.shape)

        result = cv2.warpPerspective(
            image_a, self.saved_homo_matrix, output_shape)

        result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b
        # Return the stitched image
        return result

    @staticmethod
    def detect_and_extract(image):
        # Detect and extract features from the image (DoG keypoint detector and SIFT feature extractor)
        descriptor = cv2.xfeatures2d.SIFT_create()

        (keypoints, features) = descriptor.detectAndCompute(image, None)

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
        if len(matches) > minMatches:
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
        print(n_frames)
        print(fps)
        frames = []

        for _ in tqdm.tqdm(np.arange(n_frames)):
            # Grab the frames from their respective video streams
            ok, left = left_video.read()
            _, right = right_video.read()

            if ok:
                # Stitch the frames together to form the panorama
                stitched_frame = self.stitch([left, right])

                # No homography could not be computed
                if stitched_frame is None:
                    print("[INFO]: Homography could not be computed!")
                    break

                # Add frame to video
                # stitched_frame = imutils.resize(
                #     stitched_frame, width=stitched_frame.shape[1])

                # if needClockWiseRotation is True:
                #     stitched_frame = cv2.rotate(stitched_frame, cv2.ROTATE_90_CLOCKWISE)

                stitched_frame = trim(stitched_frame)
                frames.append(stitched_frame)

                if currentFrameShow:
                    cv2.imshow("Result", stitched_frame)

                # If the 'q' key was pressed, break from the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.destroyAllWindows()
        print('[INFO]: Video stitching finished')

        # Save video
        print('[INFO]: Saving {} in {}'.format(self.video_out_path.split('/')[-1],
                                               os.path.dirname(self.video_out_path)))

        height, width, layers = frames[0].shape

        clip = cv2.VideoWriter(self.video_out_path, cv2.VideoWriter_fourcc(*'avc1'),
                               fps, (width,height))

        for frame in frames:
            clip.write(frame)

        clip.release()
        print('[INFO]: {} saved'.format(self.video_out_path.split('/')[-1]))


def main(video1,video2):
    videoOutPath = video1.split("/")[-1].split("_")[0] + ".mp4"
    stitcher = VideoStitcher(left_video_in_path=video1,
                            right_video_in_path=video2,
                            video_out_path=videoOutPath)
    stitcher.run()
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    returnPath = os.path.join(ROOT_DIR, videoOutPath)
    return  returnPath

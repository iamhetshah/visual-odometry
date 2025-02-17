import os
import cv2
import numpy as np


class VisualOdometry:
    def __init__(self, K):
        self.K = K
        # self.trackers = cv2.ORB_create(
        #     nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
        self.trackers = cv2.SIFT_create(nfeatures=5000)
        # self.matcher = cv2.FlannBasedMatcher(
        #     dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2),
        #     dict(checks=100)
        # )
        self.matcher = cv2.BFMatcher()

        self.prev_frame_gray = None
        self.prev_kp = []
        self.prev_des = []
        self.pose = np.eye(4)
        self.trajectory = []

        self.poses = []
        self.essential_matrices = []
        self.pixel_data = []
        self.count = 0

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_kp, current_des = self.trackers.detectAndCompute(
            frame_gray, None)

        if self.prev_frame_gray is not None and current_des is not None and self.prev_des is not None:
            matches = self.matcher.knnMatch(
                self.prev_des, current_des, k=2)
            good_matches = []
            try:
                for m, n in matches:
                    if m.distance < 1 * n.distance:
                        good_matches.append(m)
            except:
                pass

            print(f"Found: {len(matches)}, Matches: {len(good_matches)}")

            pt1 = []
            pt2 = []
            if len(good_matches) > 0:
                # for m, n in good_matches:
                pt1 = np.float32(
                    [self.prev_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pt2 = np.float32(
                    [current_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # pt1 = np.array(pt1)
                # pt2 = np.array(pt2)

                E, mask = cv2.findEssentialMat(
                    pt1, pt2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1)

                if E is not None and mask is not None and np.sum(mask) > 0:
                    _, R, t, _ = cv2.recoverPose(E, pt1, pt2, self.K)

                    current_transformation = np.eye(4)
                    current_transformation[:3, :3] = R
                    current_transformation[:3, 3] = t.reshape(3)
                    self.pose = self.pose @ current_transformation

                    self.poses.append(self.pose.copy())
                    self.essential_matrices.append(
                        E.copy())

                    current_position = self.pose[:3, 3]
                    self.trajectory.append(current_position)

                    self.visualize_tracking(frame, current_kp, matches)

        self.prev_frame_gray = frame_gray
        self.prev_kp = current_kp
        self.prev_des = current_des

    def visualize_tracking(self, frame, keypoints, matches):
        mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        try:
            for m, n in matches:
                if m.distance < .4 * n.distance:
                    pt1 = tuple(map(int, self.prev_kp[m.queryIdx].pt))
                    pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
                    cv2.circle(mask, pt2, radius=1, color=(
                        255, 0, 0), thickness=10)
                    cv2.line(mask, pt1, pt2, (0, 0, 255),
                             thickness=3)

            img = cv2.add(frame, mask)
            self.count += 1
            cv2.imwrite(os.path.join(self.save_path, f"{self.count}.jpg"), img)
            cv2.imshow('Tracking', img)
            cv2.waitKey(1)
        except:
            pass

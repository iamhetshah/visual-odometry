import cv2
import numpy as np
import glob


def get_calibration_data(image_path, checkerboard=(6, 6), limit=-1):
    CHECKERBOARD = checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = []
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                              0:CHECKERBOARD[1]].T.reshape(-1, 2)
    images = glob.glob(image_path)
    counter = 0
    for fname in images:
        if limit > 0 and (counter == limit):
            break
        counter += 1
        print("Fetching world data...image ", counter)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD)

        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            # resized = cv2.resize(img, (899, 1199),
            #                      interpolation=cv2.INTER_LINEAR)
            # cv2.imshow('img', resized)
            # cv2.waitKey(0)
    print(f'Calibrating...{len(imgpoints)} points')
    data = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)
    return data

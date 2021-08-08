import numpy as np
import cv2
import matplotlib.image as mpimg
import glob

objPoints = []
imgPoints = []

def chessCorners(img):
    """
    Find chess corners for an image
    """
    # Size of chessboard grid.
    NX = 9
    NY = 6
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    OBJP = np.zeros((NY*NX,3), np.float32)
    OBJP[:,:2] = np.mgrid[0:NX,0:NY].T.reshape(-1,2)
    # find corners
    ret, corners = cv2.findChessboardCorners(img, (NX, NY), None)
    # If found add new corners for calibration
    if ret == True:
        # Push to our object points and image points array
        objPoints.append(OBJP)
        imgPoints.append(corners)
        return True
    return False


def points_collector():
    """
    Loops through the calibration images and
    performs a camera calibration and collects image points
    from chessboard corners
    """
    images = glob.glob('camera_cal/calibration*.jpg')
    for file in images:
        img = mpimg.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        chessCorners(gray)
    print(len(imgPoints))
    return imgPoints, objPoints


def calibrate_camera(shape, imgPoints, objPoints):
    """
    Takes in the image points and object points and run camera calibration.
    Returns distortion coefficients, vectors and camera matrix.
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, shape, None, None)
    print(mtx)
    return ret, mtx, dist, rvecs, tvecs


def undistort(img, coefficients, matrix):
    """
    Undistorts an image
    """
    return cv2.undistort(img, matrix, coefficients, None, matrix)

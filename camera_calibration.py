import numpy as np
import cv2
import os
import json
from camera_intrinsics import CameraIntrinsics

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = []

dir_name = 'data/Assignment_MV_02_calibration'
for file in os.listdir(dir_name):
    if file.endswith(".png"):
        images.append(os.path.join(dir_name, file))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,5), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7,5), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

img = cv2.imread('data/Assignment_MV_02_calibration/Assignment_MV_02_calibration_7.png')
h,  w = img.shape[:2]

# Find scaled camera matrix for this image
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]

# Save result for viewing
cv2.imwrite('data/calib_results/calibresult.png',dst)

# Print calibration results
print('Camera matrix:')
print(mtx)
print('Distortion coefficients:')
print(dist)
print('Rotation vectors:')
print(rvecs)
print('Translation vectors:')
print(tvecs)

# Save camera intrinsics to a file
intrinsics = CameraIntrinsics(mtx, dist)
intrinsics.save("data/camera_intrinsics.json")

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

# Print error
print("total error: " + str(mean_error/len(objpoints)))

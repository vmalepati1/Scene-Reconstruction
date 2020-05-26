import numpy as np
import cv2
import os
import json
import pickle
from random import randrange
import math
from numpy import linalg as LA
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

class CameraIntrinsics:

    def __init__(self, K=None, d=None):
        self.K = np.array(K)
        self.dist_coeff = np.array(d)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)

        self.K = np.array(data['camera_matrix'])
        self.dist_coeff = np.array(data['dist_coeff'])

    def save(self, path):
        data = {"camera_matrix": self.K.tolist(), "dist_coeff": self.dist_coeff.tolist()}

        with open(path, "w") as f:
            json.dump(data, f)

class CameraCalibration:

    def __init__(self, dir_name, distorted_img_path, calib_results_path, intrinsics_path):
        self.dir_name = dir_name
        self.distorted_img_path = distorted_img_path
        self.calib_results_path = calib_results_path
        self.intrinsics_path = intrinsics_path

    def run(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((5*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        images = []

        for file in os.listdir(self.dir_name):
            if file.endswith(".png"):
                images.append(os.path.join(self.dir_name, file))

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

                # Draw and display the corners to subpixel accuracy (Part A)
                img = cv2.drawChessboardCorners(img, (7,5), corners2,ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)

        cv2.destroyAllWindows()

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

        img = cv2.imread(self.distorted_img_path)
        h,  w = img.shape[:2]

        # Find scaled camera matrix for this image
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]

        # Save result for viewing
        cv2.imwrite(self.calib_results_path,dst)

        # Print calibration results
        print('Camera matrix:')
        # Print calibration matrix K (Part B)
        print(mtx)
        print('Distortion coefficients:')
        print(dist)
        print('Rotation vectors:')
        print(rvecs)
        print('Translation vectors:')
        print(tvecs)

        # Save camera intrinsics to a file
        intrinsics = CameraIntrinsics(mtx, dist)
        intrinsics.save(self.intrinsics_path)

        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error

        # Print error
        print("total error: " + str(mean_error/len(objpoints)))

class OpticalFlow:
    def __init__(self, video_src, intrinsics):
        self.lk_params = dict( winSize  = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))    

        self.feature_params = dict( maxCorners = 500, 
                               qualityLevel = 0.3,
                               minDistance = 7,
                               blockSize = 7 )
        
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        # Try to track features throughout all frames
        self.track_len = int(self.cam.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.intrinsics = intrinsics

    def run(self):
        print('**Press escape after done viewing**')
        
        while True:
            ret, frame = self.cam.read()

            if ret:
                frame = cv2.undistort(frame, self.intrinsics.K, self.intrinsics.dist_coeff)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vis = frame.copy()

                if len(self.tracks) > 0:
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    # Use KLT algorithm to track features across image sequence (Part D)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                    d = abs(p0-p0r).reshape(-1, 2).max(-1)
                    good = d < 1
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                        if not good_flag:
                            continue
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                    self.tracks = new_tracks
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                    self.draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))

                if self.frame_idx % self.detect_interval == 0:
                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    # Identify good features to track in first frame (Part C)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **self.feature_params)
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            # Feature point coordinates are in sub-pixel accuracy
                            self.tracks.append([(x, y)])


                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break

        # Exclude tracks where the feature stayed in the same position for the entire duration
        self.tracks = [t for t in self.tracks if len(t) > 1]

        cv2.destroyAllWindows()

    def anorm2(self, a):
        return (a*a).sum(-1)

    def draw_str(self, dst, target, s):
        x, y = target
        cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.LINE_AA)
        cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.LINE_AA)		

class FundamentalMatrixCalculation:

    def __init__(self, tracker):
        self.klt_tracker = tracker

    def run(self):
        tracks = self.klt_tracker.tracks

        # Calculate correspondences

        # List of homogeneous vectors
        # Determines correpondence vectors (Part A)
        
        correspondence_vectors = []

        for track in tracks:
            initial_coord = np.array([track[0][0], track[0][1]])
            terminal_coord = np.array([track[-1][0], track[-1][1]])
            v = np.array(terminal_coord - initial_coord, ndmin=3)
            v = np.append(v, 1)
            
            correspondence_vectors.append(v)

        def calculate_scale_factor(x_coords, y_coords, x_mean, y_mean):
            norm = 0
            for x, y in zip(x_coords, y_coords):
                norm += math.sqrt(pow(x - x_mean, 2) + pow(y - y_mean, 2))

            n = len(x_coords)
            return norm / (n * math.sqrt(2))

        # Determine homographies (Part B)
        points = [t[0] for t in tracks]
        points_prime = [t[-1] for t in tracks]

        x_coords = [t[0][0] for t in tracks]
        y_coords = [t[0][1] for t in tracks]

        x_prime_coords = [t[-1][0] for t in tracks]
        y_prime_coords = [t[-1][1] for t in tracks]

        x_mean = np.average(x_coords)
        y_mean = np.average(y_coords)

        x_prime_mean = np.average(x_prime_coords)
        y_prime_mean = np.average(y_prime_coords)

        d = calculate_scale_factor(x_coords, y_coords, x_mean, y_mean)
        d_prime = calculate_scale_factor(x_prime_coords, y_prime_coords, x_prime_mean, y_prime_mean)

        # Transformation matrices for initial and final frame coordinates
        T = np.array([[1/d, 0, -x_mean/d],
                      [0, 1/d, -y_mean/d],
                      [0, 0, 1]])

        T_prime = np.array([[1/d_prime, 0, -x_prime_mean/d_prime],
                      [0, 1/d_prime, -y_prime_mean/d_prime],
                      [0, 0, 1]])

        print('Homographies:')
        print(T)
        print(T_prime)

        selection_results = []

        # Repeat procedure 10000 times (Part G_
        print('Running iterations')
        iterations = 10000

        for it_num in range(iterations):
            # Eight point DLT algorithm
            indices = []
            A = []

            # Construct A matrix for DLT algorithm (Part C)
            for i in range(8):
                found = False
                
                while not found:
                    rand_index = randrange(len(tracks))

                    if rand_index not in indices:
                        coord = T @ np.append(np.array(points[rand_index]), 1).reshape((3, 1))
                        coord_prime = T_prime @ np.append(np.array(points_prime[rand_index]), 1).reshape((3, 1))

                        x = coord[0].item() / coord[2].item()
                        y = coord[1].item() / coord[2].item()
                        x_prime = coord_prime[0].item() / coord_prime[2].item()
                        y_prime = coord_prime[1].item() / coord_prime[2].item()

                        A.append([x_prime * x, x_prime * y, x_prime, y_prime * x,
                                                y_prime * y, y_prime, x, y, 1])
                        
                        indices.append(rand_index)

                        found = True
                        
            A = np.array(A)

            U, S, VH = LA.svd(A)

            least_sv_index = np.argmin(S)

            # Calculate fundamental matrix using DLT algorithm (Part D)
            F_hat = VH.transpose()[:, least_sv_index]
            F_hat = np.reshape(F_hat, (3, 3))

            U, S, VH = LA.svd(F_hat)

            S = np.array([S[0], S[1], 0])

            F_hat = U @ np.diag(S) @ VH

            # Make sure F_hat is singular
            tolerance = 1e-13
            if LA.det(F_hat) > tolerance:
                print('Fundamental matrix is not singular')

            # Apploy normalization homographies to determine final fundamental matrix (Part D)
            F = T_prime.transpose() @ F_hat @ T

            num_outliers = 0
            inlier_sum = 0
            outlier_indices = []

            for i in range(len(tracks)):
                if i not in indices:
                    # For remaining correspondences, calculate the g_i its variance (Part E)
                    x_prime = np.append(np.array(points_prime[i]), 1).reshape((3, 1))
                    x = np.append(np.array(points[i]), 1).reshape((3, 1))

                    g_i = x_prime.transpose() @ F @ x

                    # print(g_i)

                    C_xx = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]])

                    # o = x_prime_T * F * C_xx * F_T * x_prime + x_T * F_T * C_xx * F * x
                    variance = x_prime.transpose() @ F @ C_xx @ F.transpose() @ x_prime + \
                                x.transpose() @ F.transpose() @ C_xx @ F @ x

                    # Calculate test statistic (Part F)
                    T_i = pow(g_i, 2) / variance

                    # Use an outlier threshold and sum up test statistics over all inliers
                    if T_i > 6.635:
                        num_outliers += 1
                        outlier_indices.append(i)
                    else:
                        inlier_sum += T_i.item()

            selection_results.append([F, num_outliers, inlier_sum, outlier_indices])

        print('Finished fundamental matrix iterations')
        
        # Select the fundamental matrix with fewest outliers and least inlier test statistic (Part G)
        selection_results = sorted(selection_results, key = lambda x: (x[1], x[2]))

        print('Fundamental matrix:')
        print(selection_results[0][0])
        print('Number of outliers:')
        print(selection_results[0][1])
        print('Inlier sum:')
        print(selection_results[0][2])
        print('Outlier indices:')
        print(selection_results[0][3])

        F = selection_results[0][0]
        num_outliers = selection_results[0][1]
        inlier_sum = selection_results[0][2]
        outlier_indices = selection_results[0][3]

        last_frame_idx = self.klt_tracker.cam.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        self.klt_tracker.cam.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)

        correspondances = [[a, b] for a, b in zip(points, points_prime)]

        print('**Press escape after done viewing**')
        while True:
            ret, frame = self.klt_tracker.cam.read()
            
            if ret:
                vis = frame.copy()

                outlier_correspondances = [correspondances[i] for i in outlier_indices]

                # Show inliers in green and outliers in red (Part H)
                cv2.polylines(vis, [np.int32(tr) for tr in correspondances], False, (0, 255, 0))
                # Overlay outliers in red
                cv2.polylines(vis, [np.int32(tr) for tr in outlier_correspondances], False, (0, 0, 255))
                
                cv2.imshow('last_frame', vis)
                
            if cv2.waitKey(0) == 27:
                break

        d, u = LA.eig(F.transpose() @ F)

        least_sv_index = np.argmin(d)

        # Calculate epipole locations (Part H)
        epipole1 = u[:, least_sv_index]
        epipole1 = epipole1 / epipole1[2]

        d, u = LA.eig(F @ F.transpose())

        least_ev_index = np.argmin(d)

        epipole2 = u[:, least_ev_index]
        epipole2 = epipole2 / epipole2[2]

        print('First frame epipole:')
        print(epipole1)
        print('Last frame epipole:')
        print(epipole2)
            
        # self.klt_tracker.cam.release()
        cv2.destroyAllWindows()

        return F, outlier_indices

class EssentialMatrixAnd3D:

    def __init__(self, F, outlier_indices, tracker, intrinsics):
        self.F = F
        self.outlier_indices = outlier_indices
        self.klt_tracker = tracker
        self.intrinsics = intrinsics
    
    def get_3d_points(self, points, points_prime, outliers, intrinsics, R, t):
        projected_points = []

        for i in range(len(points)):
            if i not in outliers:
                # Calculate for each inlier directions m and m_prime (Part C)
                m = LA.inv(intrinsics.K) @ np.append(np.array(points[i]), 1).reshape((3, 1))
                m_prime = LA.inv(intrinsics.K) @ np.append(np.array(points_prime[i]), 1).reshape((3, 1))

                # Calculate unknown distances l and u by solving linear equation (Part C)
                A = np.array([[(m.transpose() @ m).item(), (-m.transpose() @ R @ m_prime).item()],
                              [(m.transpose() @ R @ m_prime).item(), (-m_prime.transpose() @ m_prime).item()]])

                B = np.array([t.transpose() @ m, t.transpose() @ R @ m_prime]).reshape((2, 1))

                x = LA.solve(A, B)

                l = x[0]
                u = x[1]

                # Discard points which are behind either of the frames (Part C)
                if l > 0 and u > 0:
                    projected_points.append(l * m)

        C = np.zeros(3).reshape((3, 1))
        C_prime = t + R @ C

        return [len(projected_points), projected_points, C, C_prime, R, t]

    def reproject(self, points3d, points, R, t, K, d, cam):
        image_points, j = cv2.projectPoints(points3d, R, t, K, d)

        while True:
            ret, frame = cam.read()
            
            if ret:
                vis = frame.copy()
                
                for point in points:
                    cv2.circle(vis, point, 1, (0,255,0))

                for point in image_points:
                    cv2.circle(vis, tuple(np.int32(point.reshape(2))), 1, (0,0,255))

                cv2.imshow('frame', vis)
                
            if cv2.waitKey(0) == 27:
                break

    def run(self):
        # Use the fundamental matrix F and calibration matrix K to calculate E (Part A)
        E = self.intrinsics.K.transpose() @ self.F @ self.intrinsics.K

        U, S, V_T = LA.svd(E)

        # Enforce rotation matrices have positive determinants
        # and non-zero singular values are equal to 1
        if LA.det(U @ V_T) < 0:
            V = V_T.transpose()
            V_T = (-V).transpose()

        E = U @ np.diag([1, 1, 0]) @ V_T

        print('Essential matrix:')
        print(E)

        W = np.array([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])

        # Determine four potential combinations of R and t (Part B)
        R_1 = U @ W.transpose() @ V_T
        R_2 = U @ W @ V_T
        T = U[:, 2]
        T = np.reshape(T, (3, 1))

        # Given that camera was traveling at 50 km/hour and video was taken at 30fps
        actual_z_meters = (self.klt_tracker.track_len * 50 * 1000) / (30 * 3600)

        # Rescale z (Part B)
        z_scale_factor = actual_z_meters / T[2]
        T = T * z_scale_factor

        tracks = self.klt_tracker.tracks
            
        points = [t[0] for t in tracks]
        points_prime = [t[-1] for t in tracks]

        results = []
        results.append(self.get_3d_points(points, points_prime, self.outlier_indices, self.intrinsics, R_1, T))
        results.append(self.get_3d_points(points, points_prime, self.outlier_indices, self.intrinsics, R_1, -T))
        results.append(self.get_3d_points(points, points_prime, self.outlier_indices, self.intrinsics, R_2, T))
        results.append(self.get_3d_points(points, points_prime, self.outlier_indices, self.intrinsics, R_2, -T))

        # Select combination of R and t that yielded most points in front of both frames (Part C)
        solution = sorted(results, key = lambda x: x[0], reverse=True)[0]

        points_3d = np.array(solution[1])
        C = solution[2]
        C_prime = solution[3]
        R = solution[4]
        t = solution[5]

        # Create 3D plot to show the two camera centers and all 3D points (Part D)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color='red', label='3D Points')
        # Plot camera centers
        ax.scatter([C[0]], [C[1]], [C[2]], color='green', label='Initial Frame Camera')
        ax.scatter([C_prime[0]], [C_prime[1]], [C_prime[2]], color='blue', label='Final Frame Camera')
        plt.legend(loc="upper right")
        plt.show()

        # Project 3d points as well as corresponding features to visualize reprojection error (Part E)
        # Display initial frame reprojection error
        self.klt_tracker.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print('**Press escape after done viewing**')
        self.reproject(points_3d, points, np.zeros(3), np.zeros(3), self.intrinsics.K, self.intrinsics.dist_coeff, self.klt_tracker.cam)

        # Display final frame reprojection error
        last_frame_idx = self.klt_tracker.cam.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        self.klt_tracker.cam.set(cv2.CAP_PROP_POS_FRAMES, last_frame_idx)
        print('**Press escape after done viewing**')
        self.reproject(points_3d, points_prime, R, -t, self.intrinsics.K, self.intrinsics.dist_coeff, self.klt_tracker.cam)

        return E, points_3d

intrinsics_file = 'data/camera_intrinsics.json'

# Calibrate camera
calib = CameraCalibration('data/Assignment_MV_02_calibration',
                          'data/Assignment_MV_02_calibration/Assignment_MV_02_calibration_7.png',
                          'data/calib_results/calibresult.png',
                          intrinsics_file)
calib.run()

intrinsics = CameraIntrinsics()
intrinsics.load(intrinsics_file)

# Optical flow and feature tracking
klt_tracker = OpticalFlow('data/Assignment_MV_02_video.mp4', intrinsics)

klt_tracker.run()

# Calculate fundamental matrix
fund = FundamentalMatrixCalculation(klt_tracker)
F, outliers = fund.run()

# Calculate essential matrix
ess = EssentialMatrixAnd3D(F, outliers, klt_tracker, intrinsics)
ess.run()

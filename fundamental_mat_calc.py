from optical_flow import OpticalFlow
import cv2
import numpy as np
import pickle
from random import randrange
import math
from numpy import linalg as LA
from camera_intrinsics import CameraIntrinsics

class FundamentalMatrixCalculation:

    def __init__(self, tracker):
        self.klt_tracker = tracker

    def run(self):
        tracks = self.klt_tracker.tracks

        # Calculate correspondences

        # List of homogeneous vectors
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

        print('Running iterations')
        iterations = 10000

        for it_num in range(iterations):
            # Eight point DLT algorithm
            indices = []
            A = []

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

            F_hat = VH.transpose()[:, least_sv_index]
            F_hat = np.reshape(F_hat, (3, 3))

            U, S, VH = LA.svd(F_hat)

            S = np.array([S[0], S[1], 0])

            F_hat = U @ np.diag(S) @ VH

            # Determine singularity
            tolerance = 1e-13
            if LA.det(F_hat) > tolerance:
                print('Fundamental matrix is not singular')

            F = T_prime.transpose() @ F_hat @ T

            num_outliers = 0
            inlier_sum = 0
            outlier_indices = []

            for i in range(len(tracks)):
                if i not in indices:
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
                                
                    T_i = pow(g_i, 2) / variance

                    if T_i > 6.635:
                        num_outliers += 1
                        outlier_indices.append(i)
                    else:
                        inlier_sum += T_i.item()

            selection_results.append([F, num_outliers, inlier_sum, outlier_indices])

        print('Finished fundamental matrix iterations')

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

                cv2.polylines(vis, [np.int32(tr) for tr in correspondances], False, (0, 255, 0))
                # Overlay outliers in red
                cv2.polylines(vis, [np.int32(tr) for tr in outlier_correspondances], False, (0, 0, 255))
                
                cv2.imshow('last_frame', vis)
                
            if cv2.waitKey(0) == 27:
                break

        d, u = LA.eig(F.transpose() @ F)

        least_sv_index = np.argmin(d)

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

        return F

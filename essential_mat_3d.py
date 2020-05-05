import numpy as np
from camera_intrinsics import CameraIntrinsics
from numpy import linalg as LA
import cv2
from optical_flow import OpticalFlow
import pickle
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

class EssentialMatrixAnd3D:

    def __init__(self, F, tracker, intrinsics):
        self.F = F
        self.klt_tracker = tracker
        self.intrinsics = intrinsics
    
    def get_3d_points(self, points, points_prime, outliers, intrinsics, R, t):
        projected_points = []

        for i in range(len(points)):
            if i not in outliers:
                m = LA.inv(intrinsics.K) @ np.append(np.array(points[i]), 1).reshape((3, 1))
                m_prime = LA.inv(intrinsics.K) @ np.append(np.array(points_prime[i]), 1).reshape((3, 1))

                A = np.array([[(m.transpose() @ m).item(), (-m.transpose() @ R @ m_prime).item()],
                              [(m.transpose() @ R @ m_prime).item(), (-m_prime.transpose() @ m_prime).item()]])

                B = np.array([t.transpose() @ m, t.transpose() @ R @ m_prime]).reshape((2, 1))

                x = LA.solve(A, B)

                l = x[0]
                u = x[1]

                # Discard points which are behind either of the frames
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

        R_1 = U @ W.transpose() @ V_T
        R_2 = U @ W @ V_T
        T = U[:, 2]
        T = np.reshape(T, (3, 1))

        # Given that camera was traveling at 50 km/hour and video was taken at 30fps
        actual_z_meters = (self.klt_tracker.track_len * 50 * 1000) / (30 * 3600)

        # Rescale z
        z_scale_factor = actual_z_meters / T[2]
        T = T * z_scale_factor

        tracks = self.klt_tracker.tracks
            
        points = [t[0] for t in tracks]
        points_prime = [t[-1] for t in tracks]

        outlier_indices = [11, 12, 14, 17, 22, 23, 29, 31, 37, 44, 45, 57, 63, 64, 74, 83, 91, 95, 100]

        results = []
        results.append(self.get_3d_points(points, points_prime, outlier_indices, self.intrinsics, R_1, T))
        results.append(self.get_3d_points(points, points_prime, outlier_indices, self.intrinsics, R_1, -T))
        results.append(self.get_3d_points(points, points_prime, outlier_indices, self.intrinsics, R_2, T))
        results.append(self.get_3d_points(points, points_prime, outlier_indices, self.intrinsics, R_2, -T))

        solution = sorted(results, key = lambda x: x[0], reverse=True)[0]

        points_3d = np.array(solution[1])
        C = solution[2]
        C_prime = solution[3]
        R = solution[4]
        t = solution[5]

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

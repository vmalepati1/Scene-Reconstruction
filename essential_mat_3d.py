import numpy as np
from camera_intrinsics import CameraIntrinsics
from numpy import linalg as LA
import cv2
from optical_flow import OpticalFlow
import pickle
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

def get_3d_points(points, points_prime, outliers, intrinsics, R, t):
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

    return [len(projected_points), projected_points, C, C_prime]

klt_tracker = OpticalFlow('data/Assignment_MV_02_video.mp4', 'data/camera_intrinsics.json')

F = np.array([[-4.77563656e-09, -2.13233451e-05,  6.86626790e-03],
 [ 2.14168399e-05,  1.26183230e-07, -8.87982890e-03],
 [-6.85568422e-03,  8.68588993e-03,  3.33790460e-02]])

intrinsics = CameraIntrinsics()
intrinsics.load("data/camera_intrinsics.json")

E = intrinsics.K.transpose() @ F @ intrinsics.K

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
actual_z_meters = (klt_tracker.track_len * 50 * 1000) / (30 * 3600)

# Rescale z
z_scale_factor = actual_z_meters / T[2]
T = T * z_scale_factor

print(T)

with open("data/tracks.txt", "rb") as fp:
    tracks = pickle.load(fp)
    
points = [t[0] for t in tracks]
points_prime = [t[-1] for t in tracks]

outlier_indices = [11, 12, 14, 17, 22, 23, 29, 31, 37, 44, 45, 57, 63, 64, 74, 83, 91, 95, 100]

results = []
results.append(get_3d_points(points, points_prime, outlier_indices, intrinsics, R_1, T))
results.append(get_3d_points(points, points_prime, outlier_indices, intrinsics, R_1, -T))
results.append(get_3d_points(points, points_prime, outlier_indices, intrinsics, R_2, T))
results.append(get_3d_points(points, points_prime, outlier_indices, intrinsics, R_2, -T))

solution = sorted(results, key = lambda x: x[0], reverse=True)[0]

points = np.array(solution[1])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')
plt.show()

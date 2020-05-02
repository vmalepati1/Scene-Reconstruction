from optical_flow import OpticalFlow
import cv2
import numpy as np
import pickle
from random import randrange
import math
from numpy import linalg as LA

klt_tracker = OpticalFlow('data/Assignment_MV_02_video.mp4', 'data/camera_intrinsics.json')
##klt_tracker.run()
##
##with open("data/tracks.txt", "wb") as fp:
##    pickle.dump(klt_tracker.tracks, fp)

with open("data/tracks.txt", "rb") as fp:
    tracks = pickle.load(fp)

# Calculate correspondences

# List of homogeneous vectors
correspondence_vectors = []

for track in tracks:

    if len(track) > 1:
        initial_coord = np.array([track[0][0], track[0][1]])
        terminal_coord = np.array([track[-1][0], track[-1][1]])
        v = np.array(terminal_coord - initial_coord, ndmin=3)
        v = np.append(v, 1)
        
        correspondence_vectors.append(v)

    else:
        correspondence_vectors.append(np.append(np.zeros(2), 1))

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

# Eight point DLT algorithm
indices = []
A = []

for i in range(8):
    found = False
    
    while not found:
        rand_index = randrange(len(tracks))

        if len(tracks[rand_index]) > 1 and rand_index not in indices:
            coord = np.matmul(T, np.array([[x_coords[rand_index]], [y_coords[rand_index]], [1]]))
            coord_prime = np.matmul(T_prime, np.array([[x_prime_coords[rand_index]], [y_prime_coords[rand_index]], [1]]))

            x = coord[0][0]
            y = coord[1][0]
            x_prime = coord_prime[0][0]
            y_prime = coord_prime[1][0]

            A.append([x_prime * x, x_prime * y, x_prime, y_prime * x,
                                    y_prime * y, y_prime, x, y, 1])
            
            indices.append(rand_index)

            found = True

A = np.array(A)
A_T = A.transpose()

lin_sys_mat = np.matmul(A_T, A)

w, v = LA.eig(lin_sys_mat)

least_w_index = np.argwhere(w == np.amin(w))

F_hat = v[least_w_index.item()]
F_hat = np.reshape(F_hat, (3, 3))
F_hat = np.divide(F_hat, LA.norm(F_hat))

U, S, VH = LA.svd(F_hat)

S = np.array([S[0], S[1], 0])

F_hat = np.matmul(np.matmul(U, np.diag(S)), VH)

# Determine singularity
tolerance = 1e-13
if LA.det(F_hat) > tolerance:
    print('Fundamental matrix is not singular')

F = np.matmul(np.matmul(T_prime.transpose(), F_hat), T)

print('Fundamental matrix:')
print(F)

F_actual = cv2.findFundamentalMat(np.float32(points), np.float32(points_prime), method=cv2.FM_8POINT)[0]

x = np.array([x_coords[0], y_coords[0], 1])
x_prime = np.array([[x_coords[0]], [y_coords[0]], [1]])

cv2.destroyAllWindows()

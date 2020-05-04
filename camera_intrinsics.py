import json
import numpy as np

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

from camera_calibration import CameraCalibration
from optical_flow import OpticalFlow
from camera_intrinsics import CameraIntrinsics
from fundamental_mat_calc import FundamentalMatrixCalculation
from essential_mat_3d import EssentialMatrixAnd3D

intrinsics_file = 'data/camera_intrinsics.json'

# Calibrate camera
calib = CameraCalibration('data/Assignment_MV_02_calibration',
                          'data/Assignment_MV_02_calibration/Assignment_MV_02_calibration_7.png',
                          'data/calib_results/calibresult.png',
                          intrinsics_file)
calib.run()

intrinsics = CameraIntrinsics()
intrinsics.load(intrinsics_file)

klt_tracker = OpticalFlow('data/Assignment_MV_02_video.mp4', intrinsics)

klt_tracker.run()

# Calculate fundamental matrix
fund = FundamentalMatrixCalculation(klt_tracker)
F = fund.run()

ess = EssentialMatrixAnd3D(F, klt_tracker, intrinsics)
ess.run()

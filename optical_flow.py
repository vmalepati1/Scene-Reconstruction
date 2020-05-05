import numpy as np
import cv2
import json

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

if __name__ == '__main__':
    klt_tracker = OpticalFlow('data/Assignment_MV_02_video.mp4', 'data/camera_intrinsics.json')
    klt_tracker.run()

    cv2.destroyAllWindows()


import numpy as np
import cv2

class PerspectiveTransformation:
    def __init__(self, scale=1):
        self.scale = scale

        pts1 = np.float32([[1135, 1130],
                    [1722, 1130],
                    [ 890, 1295],
                    [1931, 1295]])

        w = 1000
        pts2 = np.float32([[0, 0], [w, 0],
                        [0, w], [w, w]])

        left = 2000
        top =  14000
        offsets2 = np.float32([[left,   top], [left,   top],
                            [left,   top], [left,   top]])


        self.bew_size = [(w + 2*left) // self.scale,
                    (w+top)      // self.scale] 

        self.matrix = cv2.getPerspectiveTransform(pts1, (pts2+offsets2) // scale)
        self.matrix2 = cv2.getPerspectiveTransform((pts2+offsets2) // scale, pts1)
   

    def get_bird_eye_view(self, frame):
        return cv2.warpPerspective(frame, self.matrix, self.bew_size)


    def from_bird_eye_view(self, frame, size=(2560, 1440)):
        return cv2.warpPerspective(frame, self.matrix2, size)
import numpy as np
import cv2

scale = 5

pts1 = np.float32([[1135, 1130],
                   [1722, 1130],
                   [ 890, 1295],
                   [1931, 1295]])

w = 1000
pts2 = np.float32([[0, 0], [w, 0],
                   [0, w], [w, w]])

left = 2000
top =  34000
offsets2 = np.float32([[left,   top], [left,   top],
                       [left,   top], [left,   top]])

matrix = cv2.getPerspectiveTransform(pts1, (pts2+offsets2) // scale)
matrix2 = cv2.getPerspectiveTransform((pts2+offsets2) // scale, pts1)

bew_size = [(w + 2*left) // scale,
            (w+top)      // scale] 


def get_bird_eye_view(frame):
    return cv2.warpPerspective(frame, matrix, bew_size)


def from_bird_eye_view(frame):
    return cv2.warpPerspective(frame, matrix2, (2560, 1440))
import cv2
import numpy as np

video = cv2.VideoCapture('dashcam1.webm')

if not video.isOpened():
    print("Error opening video file")

pts1 = np.float32([[941, 13], [955, 13],
                [270, 410], [1486, 410]])

d = 20000

pts2 = np.float32([[200, 0], [400, 0],
                   [200, d], [400, d]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

while True:
    ret, frame = video.read()
    if not ret:
        break

    img = frame[900:-100, 500:2500, ::-1]

    bird_eye_view = cv2.warpPerspective(img, matrix, (800, d))

    cv2.imshow('Bird eye view', bird_eye_view[-2000:,:])

    gray_bird_eye_view = cv2.cvtColor(bird_eye_view, cv2.COLOR_BGR2GRAY)
    ret, th_bev = cv2.threshold(gray_bird_eye_view, 127, 255, cv2.THRESH_BINARY)

    cv2.imshow('Bird eye view thresh', th_bev[-2000:,:])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
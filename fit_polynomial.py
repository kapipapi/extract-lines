import cv2
import numpy as np

video = cv2.VideoCapture('dashcam1.webm')
video.set(cv2.CAP_PROP_POS_FRAMES, 5_000)

if not video.isOpened():
    print("Error opening video file")

orb = cv2.ORB_create()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # crop frame
    frame = frame[900:-110, :]

    # Track features
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp, des = orb.detectAndCompute(gray, None)
    frame = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
import cv2

from lib.perspective import PerspectiveTransformation

cap = cv2.VideoCapture("dashcam1.webm")

orb = cv2.ORB_create()

prev_kp = None
prev_des = None

while True:
    ret, frame1 = cap.read()

    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    pt = PerspectiveTransformation(top=2000, left=1000)

    bew = pt.get_bird_eye_view(frame)

    kp, des = orb.detectAndCompute(bew, None)

    if prev_kp is None and prev_des is None:
        continue

    

    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break

cap.release()

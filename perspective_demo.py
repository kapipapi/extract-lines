import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats

from lib.perspective import PerspectiveTransformation

video = cv2.VideoCapture('dashcam1.webm')
video.set(cv2.CAP_PROP_POS_FRAMES, 5_000)

if not video.isOpened():
    print("Error opening video file")

plt.ion()
plt.show(block=False)

while True:
    ret, frame1 = video.read()
    if not ret:
        break

    frame = frame1[:, :, ::-1]

    frame_size = frame.shape[:2][::-1]

    pt = PerspectiveTransformation(scale=10)

    img = pt.get_bird_eye_view(frame)

    h,w,c = img.shape
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    cv2.erode(th1, np.ones((3,3), np.uint8), th1, iterations=1)

    columns_summed = np.sum(th1, axis=0)
    window_size = 50
    moving_average = np.convolve(columns_summed, np.ones(window_size)/window_size, mode='same')

    peaks, _ = find_peaks(moving_average, height=3000, distance=100)

    plt.cla()
    plt.plot(moving_average)
    for peak in peaks:
        plt.axvline(x=peak, color='r', linestyle='--')
    plt.draw()


    empty = np.zeros_like(img)
    for peak in peaks:
        x_start = peak - 50
        x_end = peak + 50
        data = th1[:, x_start:x_end]

        x, y = np.nonzero(data.T)

        if x.size == 0:
            continue

        x_values = np.linspace(0, 100, 50)
        y_values = np.polyval(np.polyfit(x, y, 1), x_values)

        points = np.column_stack((x_values, y_values)).astype(np.int32) + [x_start, 0]

        cv2.line(empty, (x_start, 0), (x_start, h), (0, 255, 0), 2)
        cv2.line(empty, (x_end, 0), (x_end, h), (0, 255, 0), 2)
        cv2.polylines(empty, [points], isClosed=False, color=(0,0,255), thickness=5)

    cv2.addWeighted(img, 1, empty, 0.8, 0, img)

    th1_rgb = cv2.cvtColor(th1, cv2.COLOR_GRAY2RGB)

    im_v = cv2.hconcat([img, th1_rgb])

    cv2.imshow('Bird eye view', im_v)

    inverted = pt.from_bird_eye_view(empty, frame_size)
    
    output = cv2.addWeighted(inverted, 0.8, frame1, 1, 0)
    cv2.imshow('overlay', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
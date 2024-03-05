import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import stats

from lib.perspective import get_bird_eye_view

video = cv2.VideoCapture('dashcam1.webm')
video.set(cv2.CAP_PROP_POS_FRAMES, 5_000)

if not video.isOpened():
    print("Error opening video file")

plt.ion()
plt.show(block=False)

while True:
    ret, frame = video.read()
    if not ret:
        break

    img = frame[:, :, ::-1]

    bird_eye_view = get_bird_eye_view(img)

    h,w,c = bird_eye_view.shape
    
    gray_bird_eye_view = cv2.cvtColor(bird_eye_view, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray_bird_eye_view, 150, 255, cv2.THRESH_BINARY)

    columns_summed = np.sum(th1, axis=0)
    window_size = 50
    moving_average = np.convolve(columns_summed, np.ones(window_size)/window_size, mode='same')

    peaks, _ = find_peaks(moving_average, height=1000, distance=50)

    plt.cla()
    plt.plot(moving_average)
    for peak in peaks:
        plt.axvline(x=peak, color='r', linestyle='--')
    plt.draw()

    for peak in peaks:
        plt.axvline(x=peak, color='r', linestyle='--')
        x_start =peak - 50
        x_end = peak + 50
        data = th1[:, x_start:x_end]

    # Fit linear regression to data
    coefficients = np.polyfit(np.arange(data.shape[0]), data[:, 0], deg=1)
    slope, intercept = coefficients

    # Plot the linear regression line
    plt.plot(np.arange(data.shape[0]), slope*np.arange(data.shape[0]) + intercept, color='g')

    # Show the plot
    plt.draw()


    cv2.imshow('Bird eye view thresh', th1[:, x_start:x_end])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
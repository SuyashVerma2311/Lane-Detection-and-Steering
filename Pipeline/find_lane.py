import numpy as np
import cv2
import time
from statistics import mean
import math
import matplotlib.pyplot as plt

def nothing(x):
    return None

cap = cv2.VideoCapture("Lane_find.mp4")
### Video Dimensions : (640, 640, 3) ###

cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.createTrackbar("y1", "Control", 370, 640, nothing)
cv2.createTrackbar("y2", "Control", 640, 640, nothing)
cv2.createTrackbar("x1", "Control", 320, 320, nothing)
cv2.createTrackbar("x2", "Control", 320, 320, nothing)
cv2.createTrackbar("Memory", "Control", 7, 20, nothing)

# Interactive mode for pyplot ON #
plt.ion()
fig = None
sections = 15
show_graph = False

while True:
    ret_, frame = cap.read()
    if not ret_ or cv2.waitKey(1) == 27:
        break

    x1 = cv2.getTrackbarPos("x1", "Control")
    y1 = cv2.getTrackbarPos("y1", "Control")
    x2 = cv2.getTrackbarPos("x2", "Control")
    y2 = cv2.getTrackbarPos("y2", "Control")
    memory = cv2.getTrackbarPos("Memory", "Control")
    pts = np.array([[320-x1,y1], [320+x1,y1], [320+x2,y2], [320-x2,y2]])
    pts1 = np.float32(pts)
    frame = cv2.polylines(frame.copy(), [pts], isClosed=True, color=(255,0,0), thickness=2)

    pts_out = np.array([[0,0], [640,0], [640,640], [0,640]], np.float32)
    matrix = cv2.getPerspectiveTransform(pts1, pts_out)
    inv_matrix = cv2.getPerspectiveTransform(pts_out, pts1)
    result = cv2.warpPerspective(frame, matrix, (640,640))

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    output = cv2.inRange(gray, 200, 255)

    output = cv2.erode(output, np.ones((5,5), np.uint8), iterations=1)
    output = cv2.dilate(output, np.ones((5,5), np.uint8), iterations=1)

    slices = []
    hist_list = []
    shape = output.shape

    for i in range(sections):
        sliced_img = output[int(shape[0]*i/sections):int(shape[0]*(i+1)/sections)]
        slices.append(sliced_img)
        hist_list.append(np.sum(sliced_img, axis=0))

    if show_graph:
        index = 0
        if fig is None:
            fig, plot_figs = plt.subplots(sections)
            print("Hey")
        for plot in plot_figs:
            plot.cla()
            plot.plot(hist_list[index])
            index += 1
        plt.show(block=False)
    else:
        # time.sleep(0.025)

    cv2.imshow("Input", frame)
    cv2.imshow("outout", output)

cap.release()
cv2.destroyAllWindows()

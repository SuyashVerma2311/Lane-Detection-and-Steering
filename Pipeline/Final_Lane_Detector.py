import numpy as np
import cv2
# import time
from statistics import mean
import math

def nothing(x):
    return None

cap = cv2.VideoCapture("Lane Detection Test Video 01.mkv")
### Video Dimensions : (720, 1280, 3) ###

lanes = {'right':[], 'left':[]}

### These lines were used to figure out the best ROI for Warping and Thresholding
cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.createTrackbar("y1", "Control", 600, 720, nothing)
cv2.createTrackbar("y2", "Control", 700, 720, nothing)
cv2.createTrackbar("x1", "Control", 183, 640, nothing)
cv2.createTrackbar("x2", "Control", 419, 640, nothing)
cv2.createTrackbar("Memory", "Control", 7, 20, nothing)

while True:
    ret_, frame = cap.read()
    if not ret_ or cv2.waitKey(1) == 27:
        break

    ### These lines were used to figure out the best ROI for Warping and Thresholding
    x1 = cv2.getTrackbarPos("x1", "Control")
    y1 = cv2.getTrackbarPos("y1", "Control")
    x2 = cv2.getTrackbarPos("x2", "Control")
    y2 = cv2.getTrackbarPos("y2", "Control")
    memory = cv2.getTrackbarPos("Memory", "Control")
    pts = np.array([[640-x1,y1], [640+x1,y1], [640+x2,y2], [640-x2,y2]])
    pts1 = np.float32(pts)
    img = cv2.polylines(frame, [pts], isClosed=True, color=(255,0,0), thickness=2)

    pts_out = np.array([[0,0], [500,0], [500,600], [0,600]], np.float32)
    # pts1 = np.array([[457,600], [823,600], [1050,700], [230, 700]], np.float32)
    matrix = cv2.getPerspectiveTransform(pts1, pts_out)
    result = cv2.warpPerspective(frame, matrix, (500,600))

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    output = cv2.inRange(gray, 200, 255)

    working_region = output[int(output.shape[0]/2):output.shape[0]]
    hist = np.sum(working_region, axis=0)
    first_half = hist[:int(len(hist)/2)]
    second_half = hist[int(len(hist)/2):]

    m1 = np.amax(first_half)
    if m1 != 0:
        max1 = list(np.where(first_half == m1)[0])
        left_lane_y = max1[int(len(max1)/2)]
        if len(lanes['left']) == memory:
            del lanes['left'][0]
        lanes['left'].append(left_lane_y)
    avg_l_y = int(mean(lanes['left']))
    cv2.line(result, (avg_l_y,0), (avg_l_y,600), (0,0,255), 5)

    m2 = np.amax(second_half)
    if m2 != 0:
        max2 = list(np.where(second_half == m2)[0])
        right_lane_y = max2[int(len(max2)/2)] + 250
        if len(lanes['right']) == memory:
            del lanes['right'][0]
        lanes['right'].append(right_lane_y)
    avg_r_y = int(mean(lanes['right']))
    cv2.line(result, (avg_r_y,0), (avg_r_y,600), (0,0,255), 5)

    mid = int((avg_l_y + avg_r_y)/2)

    temp = result.copy()
    cv2.fillConvexPoly(temp, np.array([(avg_l_y,0), (avg_l_y,600), (avg_r_y,600), (avg_r_y,0)], 'int32'), (0,255,0))
    result = cv2.addWeighted(result, 0.7, temp, 0.3, 0)

    matrix = cv2.getPerspectiveTransform(pts_out, pts1)
    final_result = cv2.warpPerspective(result, matrix, (1280,720))

    temp = cv2.cvtColor(final_result, cv2.COLOR_BGR2GRAY)
    temp = cv2.inRange(temp, 0, 0)

    p = (mid, 0) # your original point
    cv2.circle(result, p, 12, (120, 120, 120), 3)
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / ((matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2]))
    p_after = (int(px), int(py)) # after transformation

    bg = cv2.bitwise_and(frame, frame, mask=temp)
    final = cv2.add(bg, final_result)

    cv2.line(final, (640, 720), p_after, (0, 0, 100), 2)
    cv2.line(final, (640, 720), (640, int(py)), (0, 0, 100), 2)

    cv2.circle(final, (640, 720), 3, (255, 255, 0), -1) # Vehicle Center
    cv2.circle(final, p_after, 10, (200, 200, 120), -1) # Lane Center
    cv2.circle(final, (640,int(py)), 13, (255, 255, 0), 2) # Frame Center
    cv2.circle(final, (640,int(py)), 3, (255, 255, 0), -1) # Frame Center

    slope1 = math.atan2(720-py, 640-px)
    final_angle = np.pi/2 - slope1
    print(final_angle)

    final = cv2.putText(final, ("Curving Angle: " + str(round(final_angle, 4))), (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 150), 2, cv2.LINE_AA)

    cv2.imshow("Img", final)
    # time.sleep(0.05)

cap.release()
cv2.destroyAllWindows()

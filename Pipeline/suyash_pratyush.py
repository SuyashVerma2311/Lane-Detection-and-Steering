import numpy as np
import cv2
import time
import math
from statistics import mean

def nothing(x):
    return None

### Reading and writing the video ###
fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
cap = cv2.VideoCapture("Find_lane.mp4")
### Video Dimensions : (640, 640, 3) ###

### Dynamically controlling the ROI for the lanes ###
### Separate window for controlling te size of the ROI (and for changin the menmory of the progrmam (later updates)) ###
cv2.namedWindow("Control", cv2.WINDOW_NORMAL)
cv2.createTrackbar("y1", "Control", 487, 640, nothing)
cv2.createTrackbar("y2", "Control", 640, 640, nothing)
cv2.createTrackbar("x1", "Control", 320, 320, nothing)
cv2.createTrackbar("x2", "Control", 320, 320, nothing)
cv2.createTrackbar("Memory", "Control", 7, 20, nothing)

mem_l=0
mem_r=640
out=cv2.VideoWriter("output_steering.avi",fourcc,5,(int(cap.get(3)),int(cap.get(4))))

### Main Loop ###
while True:
    ### Reading the frames of the video ###
    ret_, frame = cap.read()
    if not ret_ or cv2.waitKey(1) == 27:
        break

    ### Getting the ROI section of the image ###
    x1 = cv2.getTrackbarPos("x1", "Control")
    y1 = cv2.getTrackbarPos("y1", "Control")
    x2 = cv2.getTrackbarPos("x2", "Control")
    y2 = cv2.getTrackbarPos("y2", "Control")
    memory = cv2.getTrackbarPos("Memory", "Control")
    pts = np.array([[320-x1,y1], [320+x1,y1], [320+x2,y2], [320-x2,y2]])
    pts1 = np.float32(pts)
    cv2.polylines(frame, [pts], isClosed=True, color=(255,0,0), thickness=2)
    cv2.line(frame,(int(frame.shape[1]/2),0),(int(frame.shape[1]/2),frame.shape[0]),(0,255,0),2)
    pts_out = np.array([[0,0], [640,0], [640,640], [0,640]], np.float32)

    ### Transforming ROI to 640 x 640 Output --> matrix ###
    matrix = cv2.getPerspectiveTransform(pts1, pts_out)
    result = cv2.warpPerspective(frame, matrix, (640,640))
    ### Transforming the transformed ROI from 640 x 640 bakc to its original size --> inv_matrix ###
    inv_matrix = cv2.getPerspectiveTransform(pts_out, pts1)

    #offset points
    # y_offset=460
    y_offset=400
    cv2.line(frame,(0,y_offset),(int(frame.shape[1]),y_offset),(0,128,0),2)

    ### Thresholding the image to only keep only the white lines ###
    gray_ = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    output = cv2.inRange(gray_, 200, 255)

    ### Morphological gtranformation to remove noise and enlarging the lanes ###
    output = cv2.erode(output, np.ones((5,5), np.uint8), iterations=1)
    output = cv2.dilate(output, np.ones((5,5), np.uint8), iterations=1)
    #cv2.line(frame,(0,int(frame.shape[1]/2)),(int(frame.shape[0]),int(frame.shape[1])/2),(0,128,0),2)

    ### Detecting the lines in the binary image ###
    edges = cv2.Canny(output, 200, 255)
    lines = cv2.HoughLinesP(edges,rho = 2.0,theta = 2*np.pi/180,threshold = 50,minLineLength = 75,maxLineGap = 150)
    x_min=frame.shape[1]/2
    x_max=frame.shape[1]/2

    ### Plotting the detected lines and finding the their positions in the sliding windows ###
    if lines is not None:
        ### Drawing the lines ###
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(output, (x1,y1), (x2,y2), 150, 3)
        ### detecting the leftmost and rightmost point in the detected lines in the roi window###
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_min=min(x_min,x1,x2)
            x_max=max(x_max,x1,x2)

    ### Adding the correct coordinates for lanes in sliding windows to memory ###
    if x_min>frame.shape[1]/2:#if the leftmost point detected is beyond the midline,we use the x_min stored in mem_l
        x_min=mem_l
    if x_max<frame.shape[1]/2:#if the rightmost point detected is before the midline,we use the x_max stored in mem_r
        x_max=mem_r
    print("x_min", x_min )
    print("x_max", x_max)
    #adding x_min & x_max to memory
    mem_l=x_min
    mem_r=x_max
    mid=int((x_max+x_min)/2) #mid_x that we get from detected lines
    #print("mid before ",mid)
    dev=mid-frame.shape[1]/2  #our deviation
    print("deviation is ",mid," - ",frame.shape[1]/2," = ",dev)
    #mid_st=0
    ### corrected mid that our bot should follow after adjusting deviation ###
    if dev < 0:
        mid=int(frame.shape[1]/2 + abs(dev))
    else:
        mid=int(frame.shape[1]/2 - abs(dev))
    print("final mid",mid)

    ### Drawing appropriate lines and circles on respective frames of the videos to view the lanes and turning angle ###
    cv2.line(frame,(mid,0),(mid,int(frame.shape[1])),(0,0,255),2)
    cv2.circle(frame,(mid,y_offset),5,(0,255,0),-1)
    cv2.circle(frame,(int(frame.shape[1]/2),y_offset),5,(0,255,0),-1)
    cv2.line(frame,(int(frame.shape[1]/2),int(frame.shape[0])),(mid,y_offset),(0,255,255),2)
    cv2.imshow("Out", frame)
    cv2.imshow("Output", output)
    out.write(frame)

    ### Slowing down the output stream ###
    time.sleep(0.025)

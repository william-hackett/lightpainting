'''
Final Project - tracking.py
CS 1290 Computational Photography, Brown U.


Usage-

To run tracking on green pixels, use track_green()

To run tracking on an object using YOLO, use track_yolo()

'''

import numpy as np
import cv2
import imutils


def init_yolo():
    global yolo 
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
    yolo.size = int(416)
    yolo.confidence = float(0.7)

def track_green(img):
    '''
    Returns the positions (x,y) of green pixels within the input image img
    '''
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.cvtColor(np.uint8([[[0, 255, 0]]]), cv2.COLOR_BGR2HSV)
    sensitivity = 40
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)

    # green in hsv color space
    # green_range = np.array([(green[0][0][0] - sensitivity, 100, 100),
                #    (green[0][0][0] + sensitivity, 255, 255)])
    # mask green objects
    # mask = cv2.inRange(img_hsv, green_range[0], green_range[1])
    mask = cv2.inRange(img_hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # contours, hierarchy
    items = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = items

    print("number of contours found:", len(contours))

    center = (0,0)
    if len(contours) > 0:
        contours = imutils.grab_contours(items)
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        center = np.mean(box, axis=0)
    return center  # (x,y)


def track_yolo(img):
    '''
    Returns the positions (x,y) of the bounding box around the subject
    within the input image img
    '''
    points = numpy.zeros((1, 1))
    return points

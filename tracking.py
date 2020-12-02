'''
Final Project - tracking.py
CS 1290 Computational Photography, Brown U.


Usage-

To run tracking on green pixels, use track_green()

To run tracking on an object using YOLO, use track_yolo()

'''

import numpy as np
import cv2


def track_green(img):
    '''
    Returns the positions (x,y) of green pixels within the input image img
    '''
    img_hsv = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2HSV)
    green = cv2.cvtColor(np.uint8([[[0, 255, 0]]]), cv2.COLOR_BGR2HSV)
    sensitivity = 15
    # green in hsv color space
    green_range = np.array([(green[0][0][0] - sensitivity, 100, 100),
                   (green[0][0][0] + sensitivity, 255, 255)])
    # mask green objects
    mask = cv2.inRange(img_hsv, green_range[0], green_range[1])
    # find contours of the green object
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("number of contours found:", len(contours))
    center = (0,0)
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(contour)
        print(rect)
        # box = cv2.boxPoints(rect)
        center = rect
        # im = cv2.drawContours(im, [box], 0, (0, 0, 255), 2)

        # x, y, w, h = cv2.boundingRect(contour)
    return center  # (x,y)


def track_yolo(img):
    '''
    Returns the positions (x,y) of the bounding box around the subject
    within the input image img
    '''
    points = numpy.zeros((1, 1))
    return points

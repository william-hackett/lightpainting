'''
Final Project - tracking.py
CS 1290 Computational Photography, Brown U.


Usage-

To run tracking on green pixels, use track_green()

To run tracking on an object using YOLO, use track_yolo()

'''

import numpy as np
import os.path
import cv2
import imutils
from yolo import YOLO
import subprocess

yolo, backSub = None, None

def init_yolo():
    global yolo
    config = "models/cross-hands.cfg"
    weights = "models/cross-hands.weights"
    if not(os.path.isfile(config) and os.path.isfile(weights)):
        subprocess.call("./models/download-some-models.sh", shell=True)
    yolo = YOLO(config, weights, ["hand"])
    yolo.size = int(256)
    yolo.confidence = float(0.3)

def init_motion():
    global backSub
    backSub = cv2.createBackgroundSubtractorMOG2()

def track_motion(img, num_objects):
    mask = backSub.apply(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.GaussianBlur(gray, (9,9), 0)
    mask = cv2.bilateralFilter(mask,9,75,75)
    kernel = np.ones((9, 9), np.uint8())
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, None, iterations=4)
    mask = cv2.dilate(mask, None, iterations=4)
    thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    items = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = items
    centers = []
    if len(contours) > 0:
        contours = imutils.grab_contours(items)
        contours = sorted(contours, key=cv2.contourArea)[::-1]
        for i in range(min(num_objects, len(contours))):
            contour = contours[i]
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            center = np.mean(box, axis=0)
            centers.append(center)
    if len(centers) == 0:
        centers.append((0, 0))
    return centers, thresh  # (x,y) 

def track_green(img, num_objects):
    '''
    Returns the positions (x,y) of green pixels within the input image img
    '''
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.cvtColor(np.uint8([[[0, 255, 0]]]),
                         cv2.COLOR_BGR2HSV)  # gives  60 255 255
    sensitivity = 15
    kernel = np.ones((5, 5), np.uint8())

    # green in hsv color space
    green_range = np.array([(green[0][0][0] - sensitivity, 100, 100),
                            (green[0][0][0] + sensitivity, 255, 255)])

    # mask green objects
    mask = cv2.inRange(img_hsv, green_range[0], green_range[1])
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # contours, hierarchy
    items = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = items
    centers = []
    # print("number of contours found:", len(contours))
    if len(contours) > 0:
        contours = imutils.grab_contours(items)
        # areas are sorted in increasing order, we must reverse!
        contours = sorted(contours, key=cv2.contourArea)[::-1]
        for i in range(min(num_objects, len(contours))):
            contour = contours[i]
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            center = np.mean(box, axis=0)
            centers.append(center)
    if len(centers) == 0:
        centers.append((0, 0))
    return centers  # (x,y)


def track_yolo(img):
    '''
    Returns the positions (x,y) of the bounding box around the subject
    within the input image img
    '''
    dim = (img.shape[1]//2, img.shape[0]//2)
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    width, height, inference_time, results = yolo.inference(img)
    # center = [0, 0]
    centers = []
    if len(results) > 0:
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)
            centers.append(np.asarray([cx*2, cy*2], dtype=np.float32))
    if len(results) == 0:
        centers.append((0, 0))
    return centers

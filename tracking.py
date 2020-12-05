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
import shlex

def init_yolo():
    global yolo
    config = "models/cross-hands.cfg"
    weights = "models/cross-hands.weights"
    if not(os.path.isfile(config) and os.path.isfile(weights)):
        proc1 = "wget -O " +config+" https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.cfg"
        proc2 = "wget -O " +weights+" https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights"
        subprocess.call(shlex.split(proc1))
        subprocess.call(shlex.split(proc2))
    yolo = YOLO(config, weights, ["hand"])
    yolo.size = int(256)
    yolo.confidence = float(0.3)

def track_green(img):
    '''
    Returns the positions (x,y) of green pixels within the input image img
    '''
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.cvtColor(np.uint8([[[0, 255, 0]]]), cv2.COLOR_BGR2HSV)
    sensitivity = 40
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

    #print("number of contours found:", len(contours))
    center = (0, 0)
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
    dim = (img.shape[1]//2, img.shape[0]//2)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    width, height, inference_time, results = yolo.inference(img)
    center = [0, 0]
    if len(results) > 0:
        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)
            center = np.asarray([cx*2, cy*2], dtype=np.float32)
    return center

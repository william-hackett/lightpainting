'''
Final Project - tracking.py
CS 1290 Computational Photography, Brown U.


Usage-

To run tracking on green pixels:
  python main.py --s [source video file path]



To run tracking on an object using YOLO
  python main.py --yolo (or -y) --s [source video file path]

'''
import argparse
import cv2
import numpy as np
from tracking import track_green, track_yolo


def lightpainting(method, image):
    img = image
    if method == "green":
        # need to change main function to take in an image, for now, it is hard coded
        points = track_green(image)
    elif method == "yolo":
        points = track_yolo(image)
    return points

def paint(img, points):
    # draws a straight lines on image depending on the location
    output = img
    color = (255, 255, 255)
    thickness = 5
    # we need at least 2 points
    if len(points)> 2:
        for i in range(len(points)-2):
            start_point = tuple(points[i])
            end_point = tuple(points[i+1])
            output = cv2.line(output.copy(), start_point, end_point, color, thickness) 
    return output

def parse(source, method):
    """
    reads in the video input and call tracking on each frame 
    """
    cv2.namedWindow("output")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        success, image = cap.read()
    else:
        success = False
    frames = [] #if we want to save frames as a video
    points = []
    # skip the first frame bc it's a black square
    success, image = cap.read()
    while success:
        # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        # frames.append(output)
        point = lightpainting(method, image)
        if not(np.sum(point) ==0):
            points.append(point)
        output = paint(image, points)
        cv2.imshow("output", output)
        success, image = cap.read()
        # print('Read a new frame: ', success)
        key = cv2.waitKey(20)
        if key == 27 or key == ord('q'):  # exit on ESC or q
            break
    cv2.destroyWindow("output")
    cap.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS1290 Lightpainting")
    parser.add_argument("-y", "--yolo", help="Indicates to track using YOLO. Default is green.",
                        action="store_true")
    parser.add_argument(
        "-s", "--source", help="Name of the source video with extension")
    args = vars(parser.parse_args())
    if args["yolo"]:
        method = "yolo"
    else:
        method = "green"
    source = 0
    parse(source, method)

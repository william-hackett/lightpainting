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
import math

def lightpainting(method, image):
    img = image
    if method == "green":
        # need to change main function to take in an image, for now, it is hard coded
        points = track_green(image)
    elif method == "yolo":
        points = track_yolo(image)
    return points

def get_coord_change(p1, p2):
    diff = p2-p1
    hyp = math.sqrt((diff[0]**2)+(diff[1]**2))
    sin = diff[0]/hyp
    cos = diff[1]/hyp
    tan = diff[1]/diff[0]
    color_change = ((cos)*255, (sin)*255, (tan)*255)
    return color_change

def paint(output, points):
    # draws a straight lines on image depending on the location
    color = (255, 255, 255)
    thickness = 5
    # we need at least 2 points
    if len(points)> 2:
        for i in range(len(points)-2):
            p1, p2 = points[i], points[i+1]
            color_change = get_coord_change(p1, p2)
            color2 = np.add(color,color_change)/2
            start_point = tuple(p1)
            end_point = tuple(p2)
            # simple line
            # output = cv2.line(output, start_point, end_point, color, thickness)
            # custom circle drawing function
            output = custom_line(output, p1, p2, color, color2)
            color = color2
    return output

def custom_line(output, p1, p2, c1, c2):
    # draws 100 circles between two points using a color gradient
    points_on_line = np.linspace(p1, p2, 100) 
    for i in range (len(points_on_line)):
        alpha = i/len(points_on_line)
        point = points_on_line[i]
        # fade_range = np.arange(0., 1, 1./4)
        strip = (np.asarray(c1)*(1-alpha) + np.asarray(c2)*(alpha))
        output = cv2.circle(output, tuple(point), 5, strip, -1)
        x, y = int(point[0]),int(point[1])
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
        # don't add if point is (0,0)
        if not(np.sum(point) == 0):
            points.append(point)
        output = paint(image, points)
        output = cv2.flip(output,1)
        cv2.imshow("output", output)
        success, image = cap.read()
        # print('Read a new frame: ', success)
        key = cv2.waitKey(20)
        if key == 27 or key == ord('q'):  # exit on ESC or q
            break
        if len(points)>25:
            points.pop(0)
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

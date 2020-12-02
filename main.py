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


def lightpainting(method):
    if method == "green":
        # need to change main function to take in an image, for now, it is hard coded
        img = "img_name.jpeg"
        points = track_green(img)
    elif method == "yolo":
        img = "img_name.jpeg"
        points = track_yolo(img)
    output = paint(img, points)
    return output

def paint(img, points):
    # draws a straight lines on image depending on the location
    for pair in points:
        start_point = pair[0]
        end_point = pair[1]
        start_point = (0, 0)
        color = (255, 255, 255)
        thickness = 9
        output = cv2.line(img, start_point, end_point, color, thickness) 
    return output

def parse(source, method):
    """
    reads in the video input and call tracking on each frame 
    """
    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(source)
    success, image = cap.read()
    frames = [] #if we want to save frames as a video
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        frame = lightpainting(method)
        frames.append(frame)
        success, image = cap.read()
        print('Read a new frame: ', success)
        count += 1
        cv2.imshow("output", frame)
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
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

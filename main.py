'''
Final Project - tracking.py
CS 1290 Computational Photography, Brown U.


Usage-

To run tracking on green pixels:
  python main.py --s [source video file path]



To run tracking on an object using YOLO
  python main.py --yolo (or -y) --s [source video file path]

'''
import numpy as np
from tracking import track_green
from tracking import track_yolo


def lightpainting(method):
    if method == "green":
        # need to change main function to take in an image, for now, it is hard coded
        img = "img_name.jpeg"
        points = track_green(img)
    elif method == "yolo":
        img = "img_name.jpeg"
        points = track_yolo(img)


def parse(source, method):
    """
    reads in the video input and call tracking on each frame 
    """
    cap = cv2.VideoCapture(source)
    success, image = cap.read()

    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        lightpainting(method)
        success, image = cap.read()
        print('Read a new frame: ', success)
        count += 1


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
    parse(source, method)

'''
Final Project - tracking.py
CS 1290 Computational Photography, Brown U.


Usage-

To run tracking on green pixels:
  python main.py



To run tracking on an object using YOLO
  python main.py --yolo (or -y)

'''
import numpy as np
from tracking import track_green
from tracking import track_yolo

def lightpainting(method):
    if method == "green":
        # need to change main function to take in an image, for now, it is hard coded
        img = "img_name.jpeg"
        points = track_green(img)
    else if method == "yolo":
        img = "img_name.jpeg"
        points = track_yolo(img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS1290 Lightpainting")
    parser.add_argument("-y", "--yolo", help="Indicates to track using YOLO. Default is green.",
        action="store_true")
    args = vars(parser.parse_args())
    if args["yolo"]:
        method = "yolo"
    else:
        method = "green"
    lightpainting(method)

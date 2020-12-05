'''
Final Project - tracking.py
CS 1290 Computational Photography, Brown U.


Usage-

To run tracking on green pixels:
    python main.py --source (or -s) <source_file_path>
    OR
    python main.py --method (or -m) green --source (or -s) <source_file_path>



To run tracking on an object using YOLO
    python main.py --method (or -m) yolo --source (or -s) <source_file_path>

'''
import argparse
import cv2
import numpy as np
from tracking import track_green, track_yolo, init_yolo
import math
from brush import hat, hat_img, radial_hat
WIDTH = 1280
HEIGHT = 720


class Painting():
    """
    INSTANCE VARIABLES:

    - self.method is the tracking method to use for lightpainting
    - self.source is the source video file name with extension
    - self.curr_frame is the current frame being painted
    - self.frames is a list of painted frames for saving a video
    - self.points is the current list of points representing the tracked path
    """

    def __init__(self, method, source):
        self.method = method
        self.source = source
        self.curr_frame = None
        self.frames = []
        self.points = []
        self.start_color = (0, 0, 255)

        if method == "yolo":
            init_yolo()

    def point_tracking(self):
        img = self.curr_frame
        if self.method == "green":
            point = track_green(img)
        elif self.method == "yolo":
            point = track_yolo(img)
        return point

    def get_coord_change(self, p1, p2):
        diff = p2-p1
        hyp = math.sqrt((diff[0]**2)+(diff[1]**2))
        sin = diff[0]/(hyp+0.01)
        cos = diff[1]/(hyp+0.01)
        tan = diff[1]/(diff[0]+0.01)
        color_change = ((cos)*255, (sin)*255, (tan)*255)
        return color_change

    def rainbow_loop(self, color):
        b = color[0]
        g = color[1]
        r = color[2]
        if (r == 255) and (r > g):
            g += 1
        elif (g == 255) and (r > 0):
            r -= 1
        elif (g == 255) and (g > b):
            b += 1
        elif (b == 255) and (g > 0):
            g -= 1
        elif (b == 255) and (b > r):
            r += 1
        elif (r == 255) and (b > 0):
            b -= 1
        return (b, g, r)

    def paint(self, color):
        # draws a straight lines on image depending on the location
        # color = (255, 255, 255)
        # For rainbow_loop, set initial color
        # color = self.start_color
        thickness = 5
        output = self.curr_frame
        # we need at least 2 points
        if len(self.points) > 2:
            for i in range(len(self.points)-2):
                p1, p2 = self.points[i], self.points[i+1]
                #color_change = get_coord_change(p1, p2)
                #color2 = np.add(color, color_change)/2
                # For rainbow_loop, set color2 to next color in spectrum
                color2 = self.rainbow_loop(color)
                # print("Color 1: {}".format(color))
                # print("Color 2: {}".format(color2))
                start_point = tuple(p1)
                end_point = tuple(p2)
                # simple line
                # output = cv2.line(output, start_point, end_point, color, thickness)
                # custom circle drawing function
                # output = self.custom_line(output, p1, p2, color, color2)
                output = self.custom_smooth_line(output, p1, p2)
                color = color2
            self.start_color = color2
        return output

    def custom_line(self, output, p1, p2, c1, c2):
        # draws 100 circles between two points using a color gradient
        points_on_line = np.linspace(p1, p2, 100)
        for i in range(len(points_on_line)):
            alpha = i/len(points_on_line)
            point = points_on_line[i]
            # fade_range = np.arange(0., 1, 1./4)
            strip = (np.asarray(c1)*(1-alpha) + np.asarray(c2)*(alpha))
            output = cv2.circle(output, tuple(point), 5, strip, -1)
            x, y = int(point[0]), int(point[1])
        return output

    def custom_smooth_line(self, output, p1, p2):
        points_on_line = np.linspace(p1, p2, 50)
        for i in range(len(points_on_line)-1):
            point = points_on_line[i]
            next_point = points_on_line[i + 1]

            vector1 = [1, 0]
            vector2 = next_point-point
            unit_vector_1 = vector1 / np.linalg.norm(vector1)
            unit_vector_2 = vector2 / np.linalg.norm(vector2)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            theta = np.arccos(dot_product)
            # output = hat_img(8.0, 2, theta, point)
            draw = radial_hat(8.0, 2, point)
            # draw = np.expand_dims(draw, 2)
            # backtorgb = cv2.cvtColor(draw, cv2.COLOR_GRAY2RGB)
            x, y = int(point[0]), int(point[1])
        return output

    def parse(self):
        """
        reads in the video input and calls tracking on each frame
        """
        cv2.namedWindow("output")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            success, self.curr_frame = cap.read()
        else:
            success = False
        # skip the first frame bc it's a black square
        success, self.curr_frame = cap.read()
        while success:
            # cv2.imwrite("frame%d.jpg" % count, self.curr_frame)  # save frame as JPEG file
            # frames.append(output)
            point = self.point_tracking()
            # don't add if point is (0,0)
            if not(np.sum(point) == 0):
                self.points.append(point)
            output = self.paint(self.start_color)
            output = cv2.flip(output, 1)
            cv2.imshow("output", output)
            success, self.curr_frame = cap.read()
            # print('Read a new frame: ', success)
            key = cv2.waitKey(20)
            if key == 27 or key == ord('q'):  # exit on ESC or q
                break
            if len(self.points) > 30:
                self.points.pop(0)
        cv2.destroyWindow("output")
        cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS1290 Lightpainting")
    parser.add_argument("-m", "--method", type=str, default="green",
                        help="Indicates tracking method. Default is green.")
    parser.add_argument("-s", "--source", type=str, default="",
                        help="Name of the source video with extension")
    args = vars(parser.parse_args())
    if args["method"] == "green":
        method = "green"
    elif args["method"] == "yolo":
        method = "yolo"
    else:
        print("Input --method is not a valid option." +
              " Defaulting to green tracking.")
        method = "green"
    source = args["source"]
    painter = Painting(method, source)
    painter.parse()

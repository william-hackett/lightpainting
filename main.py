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
# from brush import hat, hat_img, radial_hat
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

    def __init__(self, method, source, num_objects):
        self.method = method
        self.source = source
        self.curr_frame = None
        self.frames = []
        self.points = []
        self.start_color = (0, 0, 255)
        self.num_objects = num_objects

        if method == "yolo":
            init_yolo()

    def point_tracking(self):
        img = self.curr_frame
        if self.method == "green":
            centers = track_green(img, self.num_objects)
        elif self.method == "yolo":
            centers = track_yolo(img)
        return centers

    def rainbow_loop(self, color):
        b = color[0]
        g = color[1]
        r = color[2]
        if (r == 255) and (r > g) and (b == 0):
            g += 15
        elif (g == 255) and (r > 0):
            r -= 15
        elif (g == 255) and (g > b):
            b += 15
        elif (b == 255) and (g > 0):
            g -= 15
        elif (b == 255) and (b > r):
            r += 15
        elif (r == 255) and (b > 0):
            b -= 15
        return (b, g, r)

    def paint(self, color):
        # Draws a straight line on image depending on the location
        # For rainbow_loop, set initial color
        color = self.start_color
        output = self.curr_frame
        # we need at least 2 points
        if len(self.points) > 2:
            for i in range(len(self.points)-2):
                p1, p2 = self.points[i], self.points[i+1]
                # For rainbow_loop, set color2 to next color in spectrum
                color2 = self.rainbow_loop(color)
                # start_point = tuple(p1)
                # end_point = tuple(p2)
                # Simple line
                # output = cv2.line(output, start_point, end_point, color, thickness)
                # Custom circle drawing function
                output = self.custom_line(output, p1, p2, color, color2)
                # output = self.custom_smooth_line(output, p1, p2)
                color = color2
            self.start_color = self.rainbow_loop(self.start_color)
        return output

    def custom_line(self, output, p1, p2, c1, c2):
        # Draws circles between two points using a color gradient
        distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        # radius is between 7 and 4 depending on the distance
        thickness = int(7 - (distance*3/output.shape[1]))
        points_on_line = np.linspace(p1, p2, int(distance//2))
        for i in range(len(points_on_line)):
            alpha = i/len(points_on_line)
            point = points_on_line[i]
            strip = (np.asarray(c1)*(1-alpha) + np.asarray(c2)*(alpha))
            output = cv2.circle(output, tuple(point), thickness, strip, -1)
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
        cap = cv2.VideoCapture(self.source)
        if cap.isOpened():
            success, self.curr_frame = cap.read()
        else:
            success = False
        # skip the first frame bc it's a black square
        success, self.curr_frame = cap.read()
        while success:
            # cv2.imwrite("frame%d.jpg" % count, self.curr_frame)  # save frame as JPEG file
            # frames.append(output)
            centers = self.point_tracking()
            # don't add if point is (0,0)
            for point in centers:
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
    parser.add_argument("-s", "--source", type=str, default=0,
                        help="Name of the source video with extension")

    parser.add_argument("-o", "--objects", type=str, default="",
                        help="Number of objects to track")

    args = vars(parser.parse_args())
    if args["method"] == "green":
        method = "green"
    elif args["method"] == "yolo":
        method = "yolo"
    else:
        print("Input --method is not a valid option." +
              " Defaulting to green tracking.")
        method = "green"

    if not args["objects"]:
        num_objects = 1
    else:
        num_objects = args["objects"]
    source = args["source"]
    painter = Painting(method, source, int(num_objects))
    painter.parse()

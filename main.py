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
import time
import argparse
import cv2
import numpy as np
from tracking import track_green, track_yolo, init_yolo
import math
from itertools import repeat
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
        # Approach 3: initialize pts with dummy points at different sections of the screen
        # self.points = [[np.asarray([i*WIDTH//(num_objects),i*HEIGHT//(num_objects)], dtype=np.float32)] for i in range(num_objects)]
        self.points = [[] for i in range(num_objects)]
        self.mult = False
        self.start_color = [(0, 0, 255) for i in repeat(None, num_objects)]
        self.num_objects = num_objects
        if method == "yolo":
            init_yolo()

    def point_tracking(self):
        img = self.curr_frame
        if self.method == "green":
            centers = track_green(img, self.num_objects)
        elif self.method == "yolo":
            centers = track_yolo(img)
        # print("Tracking {} objects".format(num_objects))
        return centers

    def assign_points(self, centers):
        """
        group points corresponding to each object
        :param: centers a list of points corresponding to each object
        """
        # Approach 4: simple distance, hardcoded with only 2 trackings, no for loops
        MIN_DIST = 30  # if the centers are too close together, they are prob the same object
        if np.sum(centers[0]) != 0:
            # if we have detacted 2 objects already
            if self.mult == True:
                p1 = centers[0]
                g1 = self.points[0][-1]
                g2 = self.points[1][-1]
                distance1 = np.linalg.norm(p1-g1)
                distance2 = np.linalg.norm(p1-g2)
                if distance1 <= distance2:
                    self.points[0].append(p1)
                    if len(centers) > 1 and np.linalg.norm(centers[0] - centers[1]) > MIN_DIST:
                        self.points[1].append(centers[1])
                else:
                    self.points[1].append(p1)
                    if len(centers) > 1 and np.linalg.norm(centers[0] - centers[1]) > MIN_DIST:
                        self.points[0].append(centers[1])
            # if we haven't detacted multiple objects yet
            else:
                if len(centers) == 1:
                    self.points[0].append(centers[0])
                elif len(centers) > 1:
                    if np.linalg.norm(centers[0] - centers[1]) > MIN_DIST:
                        for i in range(len(centers)):
                            self.points[i].append(centers[i])
                        # now we can start detecting multiple objects!
                        self.mult = True

        # Approach 3: simple distance with base case, n-multitracking enabled
        # if self.mult:
        #     for i in range(len(centers)):
        #         p1 = centers[i]
        #         if np.sum(p1) == 0:
        #             break
        #         mindist_index = 0
        #         mindist = float("inf")
        #         for i in range(len(self.points)):
        #             g = self.points[i][-1]
        #             dist = math.sqrt(((p1[0] - g[0]) ** 2) + ((p1[1] - g[1]) ** 2))
        #             if dist < mindist:
        #                 mindist = dist
        #                 mindist_index = i
        #         self.points[mindist_index].append(p1)
        # if len(centers) > 1 and self.mult ==False:
        #     for i in range(len(centers)):
        #         self.points[i].append(centers[i])
        #     self.mult = True
        # # if we are tracking only one object yet, append to first group
        # if len(centers) == 1 and self.mult ==False:
        #     if np.sum(centers[0]) != 0:
        #         self.points[0].append(centers[0])

        # group_assigned = [0]*2
        # center_assigned = [0]*len(centers)

        # for i in range(len(centers)):
        #     p1 = centers[i]
        #     if not(np.sum(p1) == 0):
        #         closest_group = None
        #         # stores distance between current center and each group
        #         distance_to_group = {}
        #         for pt_group in range(self.num_objects):
        #             # if center not grouped and current group is empty, assign point to group
        #             if not self.points[pt_group] and not center_assigned[i]:
        #                 closest_group = pt_group
        #                 center_assigned[i] = 1
        #             # if center is not empty, calculate center distance from group
        #             elif len(self.points[pt_group]) > 0:
        #                 p2 = self.points[pt_group][-1]
        #                 distance = math.sqrt(
        #                     ((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        #                 distance_to_group[pt_group] = distance
        #         if closest_group is None:
        #             # assign the group with min distance to center as closest group
        #             closest_group = min(distance_to_group,
        #                                 key=distance_to_group.get)
        #         self.points[closest_group].append(p1)
        #         group_assigned[closest_group] = 1
        #         center_assigned[i] = 1
        # if np.count_nonzero(center_assigned) < len(centers):
        #     for pt_group in range(self.num_objects):
        #         if len(self.points[pt_group]) > 0 and group_assigned[pt_group] == 0:
        #             self.points[pt_group].pop(0)

        # Approach 2
        # center_assigned = [None for i in repeat(None, len(centers))]
        # distances = [[math.inf for i in repeat(None, self.num_objects)] for j in repeat(None, len(centers))]
        # min_distance = math.inf
        # for pt_group in range(self.num_objects):
        #     for c in range(len(centers)):
        #         p1 = centers[c]
        #         if not(np.sum(p1) == 0):
        #             if not self.points[pt_group]:
        #                 if not center_assigned[c]:
        #                     self.points[pt_group].append(p1)
        #                     center_assigned[c] = -1
        #                 break
        #             p2 = self.points[pt_group][-1]
        #             distance = math.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
        #             distances[c][pt_group] = distance
        #             if distance < min_distance:
        #                 min_distance = distance
        #                 center_assigned[c] = pt_group
        # for i in range(len(center_assigned)):
        #     assign = center_assigned[i]
        #     if assign is not None:
        #         if assign > -1:
        #             self.points[assign].append(centers[i])

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

    def paint(self):
        # Draws a straight line on image depending on the location
        output = self.curr_frame
        # we need at least 2 points
        for pt_group in range(self.num_objects):
            # For rainbow_loop, set initial color
            color = self.start_color[pt_group]
            if len(self.points[pt_group]) > 2:
                # start paint with 1 to avoid drawing dummy points.
                for i in range(len(self.points[pt_group])-2):
                    p1, p2 = self.points[pt_group][i], self.points[pt_group][i+1]
                    # For rainbow_loop, set color2 to next color in spectrum
                    color2 = self.rainbow_loop(color)
                    # Simple line
                    # output = cv2.line(output, start_point, end_point, color, thickness)
                    # Custom circle drawing function
                    output = self.custom_line(output, p1, p2, color, color2)
                    # output = self.custom_smooth_line(output, p1, p2)
                    color = color2
                self.start_color[pt_group] = self.rainbow_loop(
                    self.start_color[pt_group])
        return output

    def custom_line(self, output, p1, p2, c1, c2):
        # Draws circles between two points using a color gradient
        distance = np.linalg.norm(p1-p2)
        # radius is between 7 and 4 depending on the distance
        # thickness = int(7 - (distance*3/output.shape[1]))
        thickness = 5
        points_on_line = np.linspace(p1, p2, int(distance//4))
        for i in range(len(points_on_line)):
            alpha = i/len(points_on_line)
            point = points_on_line[i]
            strip = (np.asarray(c1)*(1.-alpha) + np.asarray(c2)*(alpha))
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
            # multiple objects to track, assign centers to groups
            if self.num_objects > 1:
                self.assign_points(centers)
            # single object to track
            else:
                if not(np.sum(centers) == 0):
                    self.points[0].append(centers[0])
            output = self.paint()
            output = cv2.flip(output, 1)
            cv2.imshow("output", output)
            success, self.curr_frame = cap.read()
            # print('Read a new frame: ', success)
            key = cv2.waitKey(20)
            if key == 27 or key == ord('q'):  # exit on ESC or q
                break
            for pt_group in range(self.num_objects):
                if len(self.points[pt_group]) > 20:
                    self.points[pt_group].pop(0)
        cv2.destroyWindow("output")
        cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS1290 Lightpainting")
    parser.add_argument("-m", "--method", type=str, default="green",
                        help="Indicates tracking method. Default is green.")
    parser.add_argument("-s", "--source", type=str, default=0,
                        help="Name of the source video with extension")
    parser.add_argument("-o", "--objects", type=int, default=1,
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
    source = args["source"]
    num_objects = args["objects"]
    painter = Painting(method, source, num_objects)
    painter.parse()

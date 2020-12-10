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
import os
import time
import argparse
import cv2
import numpy as np
from tracking import track_green, track_yolo, init_yolo, init_motion, track_motion
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
    - self.num_objects is the number of objects to be tracked
    - self.shift is a Boolean indicating whether to shift points by shift_vec
    - self.curr_frame is the current frame being painted
    - self.frames is a list of painted frames for saving a video
    - self.points is the current list of point lists representing tracked paths
    - self.mult is a Boolean indicating whether we are tracking multiple objs
    - self.start_color is a list of BGR tuples containing path starting colors
    """

    def __init__(self, method, source, num_objects, shift, color, save):
        # Initialize the tracking method
        self.method = method
        # Initialize the video source
        self.source = source
        # Initialize the number of objects to be tracked
        self.num_objects = num_objects
        # Initialize Boolean indicating whether to shift points at each frame
        self.shift = shift
        # Vector by which to shift each point to simulate motion to the right
        self.shift_vec = np.array([-10, 0])
        # Indicates whether to save output as video
        self.save = save
        # Holds the current frame
        self.curr_frame = None
        # Store a list of painted frames in order to save video
        self.frames = []

        # Approach 3: initialize pts with dummy points at different sections of the screen
        # self.points = [[np.asarray([i*WIDTH//(num_objects),i*HEIGHT//(num_objects)], dtype=np.float32)] for i in range(num_objects)]

        # A list of lists, where each inner list is a list of points
        # representing an object's tracked path
        self.points = [[] for i in range(num_objects)]
        # Boolean indicating whether multiple objects have been detected
        self.mult = False
        # Boolean indicating whether to use hard-coded color (when self.color
        # is True) or rainbow (when self.color is False)
        self.color = color
        # List of BGR tuples indicating the starting color for each object path
        if not color:
            self.start_color = [(0, 0, 255) for i in repeat(None, num_objects)]

        # Set up for yolo and motion tracking
        if method == "yolo":
            init_yolo()
        elif method == "motion":
            init_motion()

    def point_tracking(self):
        """
        Tracks the target object(s) in the current frame and returns center(s),
        a list of points in the form [(x,y)], representing the center of the
        bounding box around the target object
        """
        img = self.curr_frame
        # Track regions of green pixels
        if self.method == "green":
            centers = track_green(img, self.num_objects)
        # Track hands with YOLO hand detection
        elif self.method == "yolo":
            centers = track_yolo(img)
        # Track moving foreground objects with background subtraction
        elif self.method == "motion":
            centers, thresh = track_motion(img,  self.num_objects)
            # cv2.imshow("threshold view", thresh) #in order to view the masked filter
        # print("Tracking {} objects".format(num_objects))
        return centers

    def assign_points(self, centers):
        """
        Assign the newly tracked points to the corresponding path
        :param: centers, a list of points corresponding to each object
        """
        # Approach 4: simple distance, hardcoded with only 2 trackings, no for loops
        MIN_DIST = 30  # if the centers are too close together, they are probably the same object
        group_assigned = [False, False]
        # If centers is not empty
        if np.sum(centers[0]) != 0:
            # If we have detected 2 objects already
            if self.mult == True:
                p1 = centers[0]
                # If the first path contains a point
                if len(self.points[0]) > 0:
                    # The last point in the first path
                    g1 = self.points[0][-1]
                    # Calculate the distance between the newly tracked point
                    # and the last point in the first path
                    distance1 = np.linalg.norm(p1-g1)
                else:
                    distance1 = MIN_DIST + 1
                # If the second path contains a point
                if len(self.points[1]) > 0:
                    # The last point in the second path
                    g2 = self.points[1][-1]
                    # Calculate the distance between the newly tracked point
                    # and the last point in the first path
                    distance2 = np.linalg.norm(p1-g2)
                else:
                    distance2 = MIN_DIST + 1
                # If p1 is closer to g1 than g2
                if distance1 <= distance2:
                    # Append p1 to the first path
                    self.points[0].append(p1)
                    # If more than one newly tracked points, and they are
                    # at least minimum distance apart, append second newly
                    # tracked point to the second path
                    if len(centers) > 1 and np.linalg.norm(centers[0] - centers[1]) > MIN_DIST:
                        self.points[1].append(centers[1])
                        group_assigned[1] = True
                    if len(centers) == 1:
                        self.points[1].pop(0)
                # Otherwise, p1 is closer to g2 than g1
                else:
                    # Append p1 to the second path
                    self.points[1].append(p1)
                    # If more than one newly tracked points, and they are
                    # at least minimum distance apart, append second newly
                    # tracked point to the first path
                    if len(centers) > 1 and np.linalg.norm(centers[0] - centers[1]) > MIN_DIST:
                        self.points[0].append(centers[1])
                        group_assigned[0] = True
                    if len(centers) == 1:
                        self.points[0].pop(0)
            # If we haven't detected multiple objects yet
            else:
                if len(centers) == 1:
                    # Append first newly tracked point to the first path
                    self.points[0].append(centers[0])
                elif len(centers) > 1:
                    if np.linalg.norm(centers[0] - centers[1]) > MIN_DIST:
                        for i in range(len(centers)):
                            self.points[i].append(centers[i])
                            group_assigned[i] = True
                        # now we can start detecting multiple objects!
                        self.mult = True

        for i in range(2):
            # if group_assigned[i] == False and len(self.points[i]) > 0 and len(centers) == 1:
            #     self.points[i].pop(0)
            if len(self.points[i]) == 0:
                self.mult = False

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

    def rainbow_loop(self, color):
        """
        Increments or decrements the appropriate channel of a given BGR color
        tuple in order to advance colors through the hues of the rainbow
        :param: color, a BGR tuple with channel values in the range [0,255]
        """
        # Get each channel value
        b = color[0]
        g = color[1]
        r = color[2]
        # Use conditional statements to determine current color state and
        # increment or decrement to advance one color foreward in the rainbow
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
        """
        Draws paths on the current frame based on the stored point lists
        """
        # Draws a straight line on image depending on the location
        output = self.curr_frame
        # Iterate through all the object paths in self.points
        for pt_group in range(self.num_objects):
            # If hard-coding color
            if self.color:
                # Set color of line
                set_color = (208, 146, 8)
                color = set_color
            else:
                # For rainbow_loop, set initial color
                color = self.start_color[pt_group]
            # We need at least 2 points to draw a line
            if len(self.points[pt_group]) > 2:
                # Iterate through all the points in the current object's path,
                # drawing a line between each pair
                for i in range(len(self.points[pt_group])-2):
                    p1, p2 = self.points[pt_group][i], self.points[pt_group][i+1]
                    # If hard-coding color
                    if self.color:
                        # Color stays the same
                        color2 = set_color
                    else:
                        # For rainbow_loop, set color2 to next color in spectrum
                        color2 = self.rainbow_loop(color)
                    # Use custom circle drawing function if the two points are
                    # sufficiently close together. Otherwise, break path for
                    # neatness of lines on screen
                    if np.linalg.norm(p1-p2) < 250:
                        output = self.custom_line(output, p1, p2, color, color2)
                        # Set current color to the stored next color
                        color = color2
                    # If the shift flag is set, then shift all points but i+1
                    if shift:
                        self.points[pt_group][i] += self.shift_vec
                # If the shift flag is set, shift the point at i+1 as well
                if shift:
                    self.points[pt_group][i+1] += self.shift_vec
                # Advance and store starting color for current object path
                if not self.color:
                    self.start_color[pt_group] = self.rainbow_loop(
                        self.start_color[pt_group])
        return output

    def custom_line(self, output, p1, p2, c1, c2):
        """
        Draws circles between p1 and p2 using a color gradient from c1 to c2
        :param: output, the current frame
        :param: p1, the first point
        :param: p2, the second point
        :param: c1, the first color
        :param: c2, the second color

        Returns the current frame with a line drawn between p1 and p2
        """
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

    # save frames as the given filename.
    def frames_to_vid(self, filename, fps):
        frame_array = self.frames
        if self.source == 0:
            fps = 5.0
        print("Video is being saved...")
        out = cv2.VideoWriter(filename,cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_array[0].shape[1], frame_array[0].shape[0]), True)
        for frame in frame_array:
            out.write(frame)
        out.release()

    # the main repl: this function handles tracking, painting and displaying
    def parse(self):
        """
        Reads in the video input, calls tracking on each frame, and displays
        the frame with the generated lightpainting
        """
        cv2.namedWindow("output")
        cap = cv2.VideoCapture(self.source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("fps", fps)
        if cap.isOpened():
            success, self.curr_frame = cap.read()
        else:
            success = False
        # Skip the first frame because it's a black square
        success, self.curr_frame = cap.read()
        while success:
            centers = self.point_tracking()
            # If there are multiple objects to track, assign newly tracked
            # points to the appropriate object paths
            if self.num_objects > 1:
                self.assign_points(centers)
            # Otherwise, there is a single object to track and append
            else:
                # Don't add if point is (0,0)
                if not(np.sum(centers) == 0):
                    self.points[0].append(centers[0])
            # Paint the light trails
            output = self.paint()
            output = cv2.flip(output, 1)
            # Show the image output
            cv2.imshow("output", output)
            # Store output for saving video
            self.frames.append(output)
            success, self.curr_frame = cap.read()
            key = cv2.waitKey(20)
            if key == 27 or key == ord('q'):  # Exit on ESC or q
                break
            # If no new centers were found, pop a point from each path
            if np.sum(centers) == 0:
                for pt_group in range(self.num_objects):
                    # Pop as long as the path is nonempty
                    if self.points[pt_group]:
                        self.points[pt_group].pop(0)
            # If new centers were found, but the path of points has a length
            # over 30, then pop a point from that path
            else:
                for pt_group in range(self.num_objects):
                    if len(self.points[pt_group]) > 30:
                        self.points[pt_group].pop(0)
        cv2.destroyWindow("output")
        cap.release()

        if self.save:
            print("Saving painted frames as video...")
            size = (self.frames[0].shape[0], self.frames[0].shape[1])
            # create an output dir if it doesn't exist
            output_dir = 'output_videos'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            video_number = len(os.listdir(output_dir))
            video_name = os.path.join(output_dir, "output" + str(video_number) + '.mp4')
            self.frames_to_vid(video_name, fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CS1290 Lightpainting")
    parser.add_argument("-m", "--method", type=str, default="green",
                        help="Indicates tracking method. Default is green.")
    parser.add_argument("-src", "--source", type=str, default=0,
                        help="Name of the source video with extension.")
    parser.add_argument("-o", "--objects", type=int, default=1,
                        help="Number of objects to track.")
    parser.add_argument("-st", "--shift", action="store_true", help=("Shift" +
        " all drawn points to simulate motion at each frame."))
    parser.add_argument("-c", "--color", action="store_true",
                        help=("Indicates whether to set color when "
                            + "initializing Painting object."))
    parser.add_argument("-sv", "--save", action="store_true", help=("Save" +
        " drawn frames as a video in the output_videos/ directory."))

    args = vars(parser.parse_args())

    # Store input args
    if args["method"] == "green":
        method = "green"
    elif args["method"] == "yolo":
        method = "yolo"
    elif args["method"] == "motion":
        method = "motion"
    else:
        print("Input --method is not a valid option." +
              " Defaulting to green tracking.")
        method = "green"
    source = args["source"]
    num_objects = args["objects"]
    shift = args["shift"]
    color = args["color"]
    save = args["save"]


    # Create Painting object and run parse() to start lightpainting
    painter = Painting(method, source, num_objects, shift, color, save)
    painter.parse()

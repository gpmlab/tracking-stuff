import cv2
import numpy as np
from itertools import groupby

class ImageManipulator:
    def __init__(self):
        print(__name__)

    def get_lines(self, edges):
        rho = 1  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 30  # minimum number of votes (intersections in Hough grid cell)
        min_line_length = 200  # minimum number of pixels making up a line
        max_line_gap = 100  # maximum gap in pixels between connectable line segments

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line segments
        lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)
        return lines

    def find_length(self, p1, p2, digits=2):
        x = np.array(p1)
        y = np.array(p2)

        length = np.sqrt(np.sum((x - y) ** 2))
        return round(length, ndigits=digits)

    def find_slope(self, p1, p2, digits=2):

        dx = abs(p2[0] - p1[0])
        dy = abs(p2[1] - p1[1])

        if dy == 0:
            return None

        return round(dx / dy, ndigits=digits)

    def find_angle(self, p1, p2):
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        angle = angle * (180 / np.pi)

        return angle

    def draw_hough_lines(self, lines, img):
        if lines is None:
            return img

        final_lines = {}
        angles = []


        for line in lines:
            for x1, y1, x2, y2 in line:

                p1 = (x1, y1)
                p2 = (x2, y2)

                angle = abs(self.find_angle(p1, p2))

                if angle < 86 or angle > 93 or p1[0] < 80:
                    continue

                slope = self.find_slope(p1, p2)
                length = self.find_length(p1, p2)

                angles.append((p1, p2, angle, slope, length))


        print(f'lines: {len(angles)}')
        for test_angle in angles:
            print(f'length={test_angle[4]} slope={test_angle[3]} angle={test_angle[2]} x1={test_angle[0][0]} y1={test_angle[0][1]} x2={test_angle[1][0]} y2={test_angle[1][1]}')



        sorted_lines = sorted(angles, key=lambda x: x[0][0])
        grouped_linex = [list(it) for k, it in groupby(sorted_lines, key=lambda x: x[3])]

        for p1, p2, angle, slope, length in sorted_lines:
            cv2.line(img, p1, p2, (255, 0, 0), 2)

        return img

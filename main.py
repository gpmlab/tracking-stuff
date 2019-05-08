from collections import deque

import cv2
from random import randint

import imutils
import numpy as np
import time

path = 'IMG_0858.mov'
# path = 0
cap = cv2.VideoCapture(path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print('Frame count:', frame_count)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # start a bit later in the video


def draw_lines(lines, img):
    if lines is None:
        return img

    for line in lines:
        for x1, y1, x2, y2 in line:

            angle = np.arctan2(y2 - y1, x2 - x1)
            angle = angle * (180/np.pi)
            angle = abs(angle)

            if angle < 85 or angle > 95:
                continue


            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img


def resize(img, x, y):
    resized = cv2.resize(img, (x, y))
    return resized


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


if not cap.isOpened():
    print('error opening video')

colors = []
ret, frame = cap.read()

colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))

previous_frame = to_gray(frame)
orange = (39, 127, 255)

red_min_1 = np.array([0, 70, 50], np.uint8)
red_max_1 = np.array([10, 255, 255], np.uint8)

red_min_2 = np.array([170, 70, 50], np.uint8)
red_max_2 = np.array([180, 255, 255], np.uint8)

lower_white = np.array([0, 0, 0], dtype=np.uint8)
upper_white = np.array([0, 0, 255], dtype=np.uint8)

previous_ball = None
ball = None
not_balls = []
ball_path = deque(maxlen=10)
dx = 0
dy = 0


def to_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def draw_contrail(img):
    if len(ball_path) > 5:
        pts = []
        for bp in ball_path:
            px = bp[0]
            py = bp[1]
            pts.append([px, py])

        cv2.polylines(img, [np.asarray(pts, np.int32)], False, (255, 255, 0), 2)

    return img


while cap.isOpened():

    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame = resize(frame, 1280, 720)

    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.addWeighted(blur, 1.5, frame, -0.5, 0)
    hsv = to_hsv(frame)

    edges = cv2.Canny(hsv, 50, 150, apertureSize=3)
    cv2.imshow('canny', edges)
    edge_dilate = cv2.dilate(edges, None, iterations=4)
    cv2.imshow('canny_dialate', edge_dilate)

    edges = cv2.Canny(edge_dilate, 100, 150, apertureSize=3)
    cv2.imshow('canny after dilate', edges)

    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 400  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments
    line_image = np.copy(frame) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # test_mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.inRange(hsv, red_min_1, red_max_1)
    mask2 = cv2.inRange(hsv, red_min_2, red_max_2)
    hsv = mask | mask2
    # cv2.imshow('filtered on color', hsv)
    hsv = cv2.dilate(hsv, None, iterations=4)
    # cv2.imshow('dilated', hsv)

    hsv = cv2.morphologyEx(hsv, cv2.MORPH_CLOSE, None)
    gray = hsv  # to_gray(hsv)
    # test = gray - previous_frame
    previous_frame = gray

    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        ball = None
        i = 0
        for c in cnts:
            area = cv2.contourArea(c)
            isConvex = cv2.isContourConvex(c)
            if not area or area < 1000:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            computed_area = np.pi * radius * radius

            if radius < 35 or radius > 72:
                continue

            x = int(x)
            y = int(y)
            radius = int(radius)
            area = int(area)

            if x == 648 or x == 647 or x == 866 or x == 649 or radius == 71:
                continue

            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            cv2.putText(frame,
                        f'x={x} y={y} r={radius} a={area}',
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        .5,
                        (0, 255, 255),
                        1)

            ball = (x, y, radius, area)
            ball_path.append(ball)

    if not ball and previous_ball:
        # estimate ball position based on previous knowledge
        print('estimating position')
        x = previous_ball[0]
        y = previous_ball[1]

        x = x + dx
        y = y + dy

        cv2.circle(frame, (x, y), previous_ball[2], (0, 0, 255, 128), 4)
        previous_ball = (x, y, previous_ball[2], previous_ball[3])

    elif ball and previous_ball:
        dx = ball[0] - previous_ball[0]
        dy = ball[1] - previous_ball[1]
        if dx != 0:
            m = dy / dx
            b = ball[1] - m * ball[0]

            if dx < 0:
                factor = -50
            else:
                factor = 50

            px = int(ball[0] + factor)
            py = int(px * m + b)

            bx = ball[0]
            by = ball[1]

            cv2.arrowedLine(frame, (ball[0], ball[1]), (px, py), (255, 0, 0), 5)

    if ball:
        previous_ball = ball

    draw_contrail(frame)

    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.putText(frame, f'frame={frame_num}',
                (0, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                .5,
                (0, 255, 255),
                1)

    draw_lines(lines, frame)

    cv2.imshow('live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = time.time() - start
    print(f'elapsed: {round(elapsed * 1000, 4)}ms')

cap.release()
cv2.destroyAllWindows()

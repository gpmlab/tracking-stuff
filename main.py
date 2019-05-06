from collections import deque

import cv2
from random import randint

import imutils
import numpy as np
import time

path = '/home/kbooth/my-stuff/tracking-stuff/test.mov'
# path = 0
cap = cv2.VideoCapture(path)

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame count:', frame_count)
cap.set(cv2.CAP_PROP_POS_FRAMES, 230)  # start a bit later in the video


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
ORANGE_MIN = np.array([5, 50, 50], np.uint8)
ORANGE_MAX = np.array([15, 255, 255], np.uint8)

previous_ball = None
ball = None
not_balls = []
ball_path = deque(maxlen=50)
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
    hsv = cv2.inRange(hsv, ORANGE_MIN, ORANGE_MAX)
    # cv2.imshow('filtered', filtered)
    # filtered = cv2.dilate(hsv, None, iterations=4)
    # cv2.imshow('dilated', filtered)

    # opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, None)
    # cv2.imshow('opening', opening)
    # gray = filtered
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
            if not area or area < 400:
                continue

            (x, y), radius = cv2.minEnclosingCircle(c)
            computed_area = 3.141 * radius * radius

            if radius < 14 or radius > 20 or y < 90:
                continue

            x = int(x)
            y = int(y)
            radius = int(radius)
            area = int(area)

            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            cv2.putText(frame,
                        f'x={x} y={y} r={radius} a={area}',
                        (int(x),
                         int(y)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.0,
                        (0, 255, 255),
                        2)

            ball = (x, y, radius, area)
            ball_path.append(ball)

    if not ball and previous_ball:
        # estimate ball position based on previous knowledge
        print('estimating position')
        x = previous_ball[0]
        y = previous_ball[1]

        x = x + dx
        y = y + dy

        cv2.circle(frame, (x, y), previous_ball[2], (0, 0, 255), 4)
        previous_ball = (x, y, previous_ball[2], previous_ball[3])

    elif ball and previous_ball:
        dx = ball[0] - previous_ball[0]
        dy = ball[1] - previous_ball[1]
        if dx != 0:
            m = dy / dx
            b = ball[1] - m * ball[0]

            px = int(ball[0] + 30)
            py = int(px * m + b)

            cv2.arrowedLine(frame, (ball[0], ball[1]), (px, py), (255, 0, 0), 5)

    if ball:
        previous_ball = ball

    draw_contrail(frame)

    cv2.imshow('computed', gray)
    cv2.imshow('live', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed = time.time() - start
    print(f'elapsed: {round(elapsed, 4) * 1000}ms')

cap.release()

cv2.destroyAllWindows()

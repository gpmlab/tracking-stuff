from collections import deque
import numpy as np


class Point:
    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value


class Line:
    def __init__(self, p1: Point, p2: Point):
        self._p1 = p1
        self._p2 = p2
        self._angle = self.get_angle()
        self._slope = self.get_slope()
        self._length = self.get_length()

    @property
    def p1(self):
        return self._p1

    @p1.setter
    def p1(self, value):
        self._p1 = value

    @property
    def p2(self):
        return self._p2

    @p2.setter
    def p2(self, value):
        self._p2 = value

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value

    @property
    def slope(self):
        return self._slope

    @slope.setter
    def slope(self, value):
        self._slope = value

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    def get_length(self, ndigits=2):
        x = np.array((self.p1.x, self.p1.y))
        y = np.array(self.p2.x, self.p2.y)

        length = np.sqrt(np.sum((x - y) ** 2))
        return round(length, ndigits=ndigits)

    def get_slope(self, ndigits=2):
        dx = abs(self.p2.x - self.p1.x)
        dy = abs(self.p2.y - self.p1.y)

        if dy == 0:
            return None

        return round(dx / dy, ndigits=ndigits)

    def get_angle(self):
        angle = np.arctan2(self.p2.y - self.p1.y, self.p2.x - self.p1.x)
        angle = angle * (180 / np.pi)

        return angle


class Ball:
    def __init__(self):
        self._location_history: deque = deque(maxlen=15)
        self._location: Point = None
        self._prev_location: Point = None
        self._velocity = None
        self._b = None

    def update_location(self, point: Point):
        if self._location:
            self._location_history.append(self._location)

        self._prev_location = self.location
        self._location = point

        if self._location and self._prev_location:
            dx = self._location.x - self._prev_location.x
            dy = self._location.y - self._prev_location.y
            if dx != 0:
                self._velocity = dy / dx
                self._b = self._location.y - (self._velocity * self._location.x)

                # if dx < 0:
                #    factor = -50
                # else:
                #    factor = 50
                # cv2.arrowedLine(frame, (ball[0], ball[1]), (px, py), (255, 0, 0), 5)

    def get_location_history(self):
        return self._location_history

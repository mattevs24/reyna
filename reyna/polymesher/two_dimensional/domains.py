"""
This file is filled with predefined/common domains for use. For more information on the Domain object, see the
documentation in '_auxilliaries/abstraction' for Domain.
"""

import math
import typing

import numpy as np

import reyna.polymesher.two_dimensional._auxilliaries.distance_functions as pmdf
from reyna.polymesher.two_dimensional._auxilliaries.abstraction import Domain


class CircleDomain(Domain):

    def area(self):
        return math.pi * (self.radius ** 2)

    def distances(self, points: np.ndarray) -> np.ndarray:
        d = pmdf.d_sphere(points, self.center, radius=self.radius)
        return d

    def pFix(self) -> typing.Optional[np.ndarray]:
        return self.fixed_points

    def __init__(self, bounding_box=np.array([[-1, 1], [-1, 1]]), fixed_points=None):
        super().__init__(bounding_box=bounding_box, fixed_points=fixed_points)
        self.center = 0.5 * np.array([[self.bounding_box[0, 1] + self.bounding_box[0, 0],
                                       self.bounding_box[1, 1] + self.bounding_box[1, 0]]])
        self.radius = 0.5 * min(self.bounding_box[0, 1] - self.bounding_box[0, 0],
                                self.bounding_box[1, 1] - self.bounding_box[1, 0])


class RectangleDomain(Domain):

    def area(self):
        return (self.bounding_box[0, 1] - self.bounding_box[0, 0]) * (self.bounding_box[1, 1] - self.bounding_box[1, 0])

    def distances(self, points: np.ndarray) -> np.ndarray:
        d = pmdf.d_rectangle(points, self.bounding_box[0, 0], self.bounding_box[0, 1],
                             self.bounding_box[1, 0], self.bounding_box[1, 1])
        return d

    def pFix(self) -> typing.Optional[np.ndarray]:
        return self.fixed_points

    def __init__(self, bounding_box, fixed_points=None):
        super().__init__(bounding_box=bounding_box, fixed_points=fixed_points)


class LShapeDomain(Domain):

    def area(self):
        rect = (self.bounding_box[0, 1] - self.bounding_box[0, 0]) * (self.bounding_box[1, 1] - self.bounding_box[1, 0])
        return 0.75 * rect

    def distances(self, points: np.ndarray) -> np.ndarray:
        b_box = self.bounding_box
        x_m = 0.5 * b_box[0, :].sum()
        y_m = 0.5 * b_box[1, :].sum()
        d_1 = pmdf.d_rectangle(points, b_box[0, 0], b_box[0, 1], b_box[1, 0], b_box[1, 1])
        d_2 = pmdf.d_rectangle(points, x_m, b_box[0, 1], b_box[1, 0], y_m)
        d = pmdf.d_difference(d_1, d_2)
        return d

    def pFix(self) -> typing.Optional[np.ndarray]:
        b_box = self.bounding_box
        x_m = 0.5 * b_box[0, :].sum()
        y_m = 0.5 * b_box[1, :].sum()
        h = 0.005
        f_ps = np.array([[x_m - h, y_m - h], [x_m + h, y_m + h], [x_m - h, y_m + h]])
        return f_ps

    def __init__(self):
        super().__init__(bounding_box=np.array([[0, 1], [0, 1]]))


class RectangleCircleDomain(Domain):

    def area(self):
        rect = (self.bounding_box[0, 1] - self.bounding_box[0, 0]) * (self.bounding_box[1, 1] - self.bounding_box[1, 0])
        radius = 0.5 * min(self.bounding_box[0, 1] - self.bounding_box[0, 0],
                           self.bounding_box[1, 1] - self.bounding_box[1, 0])
        circ = math.pi * (radius ** 2)
        return rect - circ

    def distances(self, points: np.ndarray) -> np.ndarray:
        b_box = self.bounding_box
        d_1 = pmdf.d_rectangle(points, b_box[0, 0], b_box[0, 1], b_box[1, 0], b_box[1, 1])
        d_2 = pmdf.d_sphere(points)
        d = pmdf.d_difference(d_1, d_2)
        return d

    def pFix(self) -> typing.Optional[np.ndarray]:
        return None

    def __init__(self):
        super().__init__(bounding_box=np.array([[-2, 2], [-2, 2]]))


class HornDomain(Domain):

    def area(self):
        raise NotImplementedError

    def distances(self, points: np.ndarray) -> np.ndarray:
        d_1 = pmdf.d_sphere(points)
        d_2 = pmdf.d_sphere(points, center=np.array([-0.4, 0]), radius=0.55)
        d_3 = pmdf.d_line(points, 0, 0, 1, 0)
        d = pmdf.d_intersect(d_3, pmdf.d_difference(d_1, d_2))
        return d

    def pFix(self) -> typing.Optional[np.ndarray]:
        return None

    def __init__(self):
        super().__init__(bounding_box=np.array([[-1, 1], [0, 1]]))


class CircleCircleDomain(Domain):

    def area(self):
        raise NotImplementedError

    def distances(self, points: np.ndarray) -> np.ndarray:
        d_1 = pmdf.d_sphere(points)
        d_2 = pmdf.d_sphere(points, center=np.array([0.2, 0]), radius=0.5)
        d = pmdf.d_difference(d_1, d_2)
        return d

    def pFix(self) -> typing.Optional[np.ndarray]:
        return None

    def __init__(self):
        super().__init__(bounding_box=np.array([[-1, 1], [-1, 1]]))

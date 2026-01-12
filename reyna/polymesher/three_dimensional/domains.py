import typing

import numpy as np

import reyna.polymesher.three_dimensional._auxilliaries.distance_functions as pmdf3d
from reyna.polymesher.three_dimensional._auxilliaries.abstraction import Domain3D


class CuboidDomain(Domain3D):

    def volume(self):
        return (self.bounding_box[0, 1] - self.bounding_box[0, 0]) * \
               (self.bounding_box[1, 1] - self.bounding_box[1, 0]) * \
               (self.bounding_box[2, 1] - self.bounding_box[2, 0])

    def distances(self, points: np.ndarray) -> np.ndarray:
        d = pmdf3d.d_cuboid(points, self.bounding_box)
        return d

    def pFix(self) -> typing.Optional[np.ndarray]:
        return self.fixed_points

    def boundary_conditions(self, **kwargs):
        raise NotImplementedError("This is not a standard method for the CuboidDomain class and is"
                                  " therefore not implemented")

    def __init__(self, bounding_box, fixed_points=None):
        super().__init__(bounding_box=bounding_box, fixed_points=fixed_points)

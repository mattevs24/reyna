import numpy as np
import typing
from abc import ABCMeta, abstractmethod
from enum import Enum, unique


class Domain2D(metaclass=ABCMeta):
    def __init__(self, bounding_box: np.ndarray, fixed_points: typing.Optional[np.ndarray] = None):
        self.bounding_box = bounding_box
        self.fixed_points = fixed_points

    @abstractmethod
    def distances(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def pFix(self) -> typing.Optional[np.ndarray]:
        pass

    @abstractmethod
    def boundary_conditions(self, **kwargs):
        pass

    @abstractmethod
    def area(self):
        pass


class Domain3D(metaclass=ABCMeta):
    def __init__(self, bounding_box: np.ndarray, fixed_points: typing.Optional[np.ndarray] = None):
        self.bounding_box = bounding_box
        self.fixed_points = fixed_points

    @abstractmethod
    def distances(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def pFix(self) -> typing.Optional[np.ndarray]:
        pass

    @abstractmethod
    def boundary_conditions(self, **kwargs):
        pass

    @abstractmethod
    def volume(self):
        pass


@unique
class Domain(Enum):
    two_d = Domain2D
    three_d = Domain3D


class Domain(metaclass=ABCMeta):
    """
    Foundational class for domain generation. Every (custom) domain must be built on this ABC.
    """
    def __init__(self, bounding_box: np.ndarray, fixed_points: typing.Optional[np.ndarray] = None):
        self.bounding_box = bounding_box
        self.fixed_points = fixed_points

    @abstractmethod
    def distances(self, points: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def pFix(self) -> typing.Optional[np.ndarray]:
        pass

    @abstractmethod
    def boundary_conditions(self, **kwargs):
        pass

    @abstractmethod
    def area(self):
        pass


@typing.runtime_checkable
class PolyMeshProtocol(typing.Protocol):
    """
    This is the base protocol for all polygonal meshes generated in this package. There are many differnet outputs
    throughout the functions of this package and this function unites them with the minimal requirements.
    """

    vertices: np.ndarray
    filtered_regions: typing.List[list]
    filtered_points: np.ndarray
    domain: Domain

    def __init__(self, vertices, filtered_regions, filtered_points, domain):
        self.vertices = vertices
        self.filtered_regions = filtered_regions
        self.filtered_points = filtered_points
        self.domain = domain


# PolyMesh = type(PolyMeshProtocol)
"""The foundational type for this package: PolyMesh."""


class PolyMesh(metaclass=ABCMeta):
    """The foundational type for this package: PolyMesh."""

    vertices: np.ndarray
    filtered_regions: typing.List[list]
    filtered_points: np.ndarray
    domain: Domain
    ridge_points: np.ndarray

    def __init__(self, vertices, filtered_regions, filtered_points, ridge_points, domain):
        self.vertices = vertices
        self.filtered_regions = filtered_regions
        self.filtered_points = filtered_points
        self.ridge_points = ridge_points
        self.domain = domain

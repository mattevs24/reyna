import numpy as np
import typing
from abc import ABCMeta, abstractmethod


class Domain(metaclass=ABCMeta):
    """
    Foundational class for domain generation. Every (custom) domain must be built on this ABC. Defines the common
    interface for all domains.

    Attributes:
        bounding_box (np.ndarray): Bounding box of the domain.
        fixed_points (np.ndarray): Fixed points of the domain.

    Notes:
        - The variable fixed points works with 'poly_mesher' and gives a set of points that remain as fixed centres
          during LLoyds algorithm.

    """
    def __init__(self, bounding_box: np.ndarray, fixed_points: typing.Optional[np.ndarray] = None):
        self.bounding_box = bounding_box
        self.fixed_points = fixed_points

    @abstractmethod
    def distances(self, points: np.ndarray) -> np.ndarray:
        """
        The distance function for a domain. This gives some indication of 'how far' a given point is from the boundary
        of the domain. Negative values determine the inside of a domain. This explicitly defines the domain.
        Args:
            points (np.ndarray): Points to calculate the distances from.

        Returns:
            np.ndarray: Distances calculated from the points.
        """
        pass

    @abstractmethod
    def pFix(self) -> typing.Optional[np.ndarray]:
        """
        This function returns a set of fixed points. This 'getter' is used if the set of fixed points is in fact
        dependent on additional domain parameters. If no more points are required, then use 'return self.fixed_points'.

        Returns:
            typing.Optional[np.ndarray]: Array of fixed points.
        """
        pass

    @abstractmethod
    def area(self) -> float:
        """
        This function returns the area of a domain.
        Returns:
            float: Area of the domain.
        """
        pass


class PolyMesh(metaclass=ABCMeta):
    """
    This ABC defines the PolyMesh class for this package. This may be used by the user to generate meshes not availible
    with the standard set to be able to be used with the numerical methods of this package.

    Attributes:
        vertices (np.ndarray): Array of vertices.
        filtered_regions: Elements of the mesh (list of lists of integer indecies to 'vertices')
        filtered_points: The corresponding centers of these elements.
        domain (Domain): The domain class used to generate this PolyMesh.

    Notes:
        - The 'filtered_points" amy be any points so long as they lie in the kernel of the element (e.g. Voronoi centers
        if the mesh is Voronoi). I.e. the elements must be relatively convex with respect to this point.

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

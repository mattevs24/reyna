import numpy as np
import typing
from abc import ABCMeta, abstractmethod


class Domain3D(metaclass=ABCMeta):
    """
    Foundational class for domain generation. Every (custom) domain must be built on this ABC to function correctly
    with the poly_mesher function. Defines the common interface for all domains.

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
    def volume(self) -> float:
        """
        This function returns the area of a domain.

        Returns:
            float: Area of the domain.
        """
        pass


class PolyMesh3D(metaclass=ABCMeta):
    """
    This ABC defines the PolyMesh class for this package. This may be used by the user to generate meshes not availible
    with the standard set to be able to be used with the numerical methods of this package.

    Attributes:
        vertices (np.ndarray): Array of vertices.
        facets (typing.List[list]): Facets of the mesh (list of lists of integer indecies to 'vertices')
        elements (typing.List[list]): Elements of the mesh (list of lists of integer indecies to 'vertices')
        centers (np.ndarray): The corresponding centers of these elements.
        domain (Domain): The domain class used to generate this PolyMesh.

    Notes:
        - The 'filtered_points" may be any points so long as they lie in the kernel of the element (e.g. Voronoi centers
          if the mesh is Voronoi). I.e. the elements must be relatively convex with respect to this point.

    """

    def __init__(self, vertices, filtered_facets, filtered_regions, facet_types, ridge_points, filtered_points, domain):
        self.vertices = vertices
        self.facets = filtered_facets
        self.elements = filtered_regions
        self.facet_types = facet_types
        self.ridge_points = ridge_points
        self.centers = filtered_points
        self.domain = domain

        # TODO: need to update this here based on what I want/need for the DGFEM Geometry and solver
        # Ideally this contains all the facet information too with indecies on the global coordinate system.
        # These facet indecies will be ordered either clockwise or anticlockwise to fit the pattern of the two
        # dimensional facets -- to do this, can cycle through ridge_vertices? this may be an exact list of facets?
        # hopefully these facet indecies will be in order as two dimensions.

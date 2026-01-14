import time
import typing

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, find
from shapely import Polygon, Point

from reyna.polymesher.three_dimensional._auxilliaries.abstraction import PolyMesh3D

from dataclasses import dataclass
import pickle


@dataclass
class DGFEMGeometry:
    """
    This is a geometry function which provides all the additional mesh information that a DGFEM method requires to
    run; normal vectors, subtriagulations/simplicial decompositions etc.
    """

    # TODO: this will be a lot more complicated than the two dimensional version. For example here, we need information
    #  on all the facets as well as the elements themselves. This will make this function more costly but hopefully it
    #  won't be prhibitive as it wasn't up to a lot of elements in the two dimensional case.

    def __init__(self, poly_mesh: PolyMesh3D, **kwargs):
        """
        Initializes DGFEMGeometry with the given PolyMesh object.

        Args:
            poly_mesh (PolyMesh): The polygonal mesh of which to generate the information.
            **kwargs: Additional keyword arguments. 'time' is the only current option and is used to time the geometry
            generation.
        """
        self.mesh = poly_mesh

        self.n_elements = len(poly_mesh.elements)
        self.n_nodes = poly_mesh.vertices.shape[0]
        self.dimension = poly_mesh.vertices.shape[1]

        self.nodes = poly_mesh.vertices

        self.elem_bounding_boxes = None  # yes -- done -- checked

        # TODO: may convert bounding boxes here to midpoints and radii? m and h's

        self.boundary_facets = None  # yes -- done
        # self.boundary_facets_to_element = None  # yes-- done

        self.interior_facets = None  # yes -- done
        # self.interior_facets_to_element = None  # yes-- done
        #
        # self.boundary_normals = None  # yes -- done
        # self.interior_normals = None  # yes -- done

        self.simplicial_decomposition = None  # yes -- done -- checked
        self.facet_subtriangulations = None  # yes -- done -- checked
        self.n_simplicies = None  # yes -- done -- checked
        self.simplex_to_element = None  # yes -- done -- checked
        self.triangle_to_facet = None   # yes? -- done -- checked

        self.h: typing.Optional[float] = None  # yes
        self.h_s: typing.Optional[np.ndarray] = None  # yes
        self.volumes: typing.Optional[np.ndarray] = None  # yes

        # TODO: I may need omre attributes than this but not 100% sure currently.

        time_generation = False
        if 'time' in kwargs:
            time_generation = kwargs.pop('time')

        _time = time.time()

        self._generate()

        if time_generation:
            print(f"Time taken to generate geometry: {time.time() - _time}s")

    def _generate(self, max_n: int = 100):

        self.nodes *= 1e8
        self.nodes /= 1e8

        sub_simplicial_decomp_0 = []
        sub_simplicial_decomp_1 = []
        sub_simplicial_decomp_2 = []
        sub_simplicial_decomp_3 = []

        simplex_to_element = []
        elem_bounding_boxes = []

        # Operations on the elements

        for i, element in enumerate(self.mesh.elements):

            element = np.array(element)

            # Bounding box calculation here
            elem_bounding_boxes.append([np.min(self.nodes[element, 0]), np.max(self.nodes[element, 0]),
                                        np.min(self.nodes[element, 1]), np.max(self.nodes[element, 1]),
                                        np.min(self.nodes[element, 2]), np.max(self.nodes[element, 2])])

            # Simplicial Decomposition here

            local_simplicial_decomp = Delaunay(self.nodes[element, :])
            n_simplices = local_simplicial_decomp.simplices.shape[0]

            simplex_to_element += n_simplices * [i]

            sub_simplicial_decomp_0 += list(element[local_simplicial_decomp.simplices[:, 0]])
            sub_simplicial_decomp_1 += list(element[local_simplicial_decomp.simplices[:, 1]])
            sub_simplicial_decomp_2 += list(element[local_simplicial_decomp.simplices[:, 2]])
            sub_simplicial_decomp_3 += list(element[local_simplicial_decomp.simplices[:, 3]])

            # Maximal Cell diameter calculation here? not sure how to do in 3D

            # Sorting the edges in total_edge_.... will be different here -- we have facets in self.mesh.facets?

        # Recombine all the simplicial decompositions

        self.elem_bounding_boxes = elem_bounding_boxes
        self.simplicial_decomposition = np.concatenate((np.array(sub_simplicial_decomp_0)[:, np.newaxis],
                                                        np.array(sub_simplicial_decomp_1)[:, np.newaxis],
                                                        np.array(sub_simplicial_decomp_2)[:, np.newaxis],
                                                        np.array(sub_simplicial_decomp_3)[:, np.newaxis]), axis=1)
        self.n_simplicies = self.simplicial_decomposition.shape[0]
        self.simplex_to_element = simplex_to_element

        # Indicate the boundary and interior facets
        self.boundary_facets = np.where(self.mesh.facet_types == 0)[0]
        self.interior_facets = np.where(self.mesh.facet_types == 1)[0]

        # # Facet to elements
        # self.boundary_facets_to_element = np.array([
        #     (self.mesh.ridge_points[i] if self.mesh.ridge_points[i, 0] < self.n_elements else self.mesh.ridge_points[
        #         i, 1]) for i in self.boundary_facets
        # ])
        #
        # self.interior_facets_to_element = self.mesh.ridge_points[self.interior_facets]

        # Operations on the elements' facets.

        sub_triangulation_0 = []
        sub_triangulation_1 = []
        sub_triangulation_2 = []

        triangle_to_facet = []

        # self.boundary_normals = np.zeros((len(self.boundary_facets), 3), dtype=float)
        # self.interior_normals = np.zeros((len(self.interior_facets), 3), dtype=float)

        for i, facet in enumerate(self.mesh.facets):

            facet = np.array(facet)
            vertices = self.nodes[facet, :]

            # Project points to 2D
            center = np.mean(vertices, axis=0)
            X = vertices - center

            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            vertices_2d = X @ Vt[:2].T  # V transpose is a basis here for the projection's codomain ([:2] as degenerate)

            normal = np.cross(Vt[0], Vt[1])

            # if self.mesh.facet_types[i] == 0:
            #     self.boundary_normals[np.argwhere(self.boundary_facets == i)[0]] = normal
            # else:
            #     self.interior_normals[np.argwhere(self.interior_facets == i)[0]] = normal

            projected_subtriangulation = Delaunay(vertices_2d)

            n_triangles = projected_subtriangulation.simplices.shape[0]
            triangle_to_facet += n_triangles * [i]

            sub_triangulation_0 += list(facet[projected_subtriangulation.simplices[:, 0]])
            sub_triangulation_1 += list(facet[projected_subtriangulation.simplices[:, 1]])
            sub_triangulation_2 += list(facet[projected_subtriangulation.simplices[:, 2]])

        self.triangle_to_facet = triangle_to_facet
        self.facet_subtriangulations = np.concatenate((np.array(sub_triangulation_0)[:, np.newaxis],
                                                       np.array(sub_triangulation_1)[:, np.newaxis],
                                                       np.array(sub_triangulation_2)[:, np.newaxis]), axis=1)


import time
import typing

import numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix, find
from shapely import Polygon, Point

from reyna.polymesher.two_dimensional._auxilliaries.abstraction import PolyMesh

from dataclasses import dataclass
import pickle


@dataclass
class DGFEMGeometry:
    """
    This is a geometry function which provides all the additional mesh information that a DGFEM method requires to
    run; normal vectors, subtriagulations etc. We do note here that this only works for convex elements. This is a
    guarentee given the PolyMesh object, but user-defined meshes may not work as required.
    """

    def __init__(self, poly_mesh: PolyMesh, **kwargs):
        """
        Args:
            poly_mesh (PolyMesh): The polygonal mesh of which to generate the information.
        """
        self.mesh = poly_mesh

        self.n_nodes = poly_mesh.vertices.shape[0]
        self.n_elements = len(poly_mesh.filtered_regions)

        # TODO - these need to go.... use the mesh object? may need to edit th emesh object though?
        self.nodes = poly_mesh.vertices

        self.elem_bounding_boxes = None
        self.n_triangles = None
        self.boundary_edges = None
        self.boundary_edges_to_element = None
        self.interior_edges = None
        self.interior_edges_to_element = None
        self.boundary_normals = None
        self.interior_normals = None

        self.subtriangulation = None
        self.triangle_to_polygon = None
        self.interior_edges_to_triangle = None
        self.boundary_edges_to_triangle = None

        self.h: typing.Optional[float] = None
        self.h_s: typing.Optional[np.ndarray] = None
        self.areas: typing.Optional[np.ndarray] = None

        time_generation = False
        if 'time' in kwargs:
            time_generation = kwargs.pop('time')

        _time = time.time()

        self.generate()

        if time_generation:
            print(f"Time taken to generate geometry: {time.time() - _time}s")

    def generate(self, max_n: int = 20):
        self.nodes *= 1e8
        self.nodes /= 1e8

        edges_per_elem = []

        subtriangulation_0 = np.zeros((max_n, self.n_elements), dtype=int) - 1
        subtriangulation_1 = np.zeros((max_n, self.n_elements), dtype=int) - 1
        subtriangulation_2 = np.zeros((max_n, self.n_elements), dtype=int) - 1

        total_edge_x = np.zeros((max_n, self.n_elements), dtype=int) - 1
        total_edge_y = np.zeros((max_n, self.n_elements), dtype=int) - 1

        tri_to_poly = []
        elem_bounding_boxes = []

        h_s = []
        areas = []

        for i, elem_i in enumerate(self.mesh.filtered_regions):

            elem_bounding_boxes.append([np.min(self.nodes[elem_i, 0]), np.max(self.nodes[elem_i, 0]),
                                        np.min(self.nodes[elem_i, 1]), np.max(self.nodes[elem_i, 1])])

            poly = Polygon(self.nodes[elem_i, :])
            box = poly.minimum_rotated_rectangle
            _x, _y = box.exterior.coords.xy
            edge_length = (Point(_x[0], _y[0]).distance(Point(_x[1], _y[1])),
                           Point(_x[1], _y[1]).distance(Point(_x[2], _y[2])))
            h_s.append(max(edge_length))
            areas.append(poly.area)

            local_sub_tri = Delaunay(self.nodes[elem_i, :])

            n_triangles = local_sub_tri.simplices.shape[0]

            tri_to_poly += n_triangles * [i]

            subtriangulation_0[:n_triangles, i] = elem_i[local_sub_tri.simplices[:, 0]]
            subtriangulation_1[:n_triangles, i] = elem_i[local_sub_tri.simplices[:, 1]]
            subtriangulation_2[:n_triangles, i] = elem_i[local_sub_tri.simplices[:, 2]]

            n_edges = len(elem_i)

            total_edge_x[:n_edges, i] = elem_i
            total_edge_y[:n_edges, i] = np.roll(elem_i, -1)

            edges_per_elem.append(n_edges)

        self.h_s = np.array(h_s)
        self.areas = np.array(areas)
        self.h = np.max(self.h_s)

        subtriangulation_0 = subtriangulation_0.flatten(order="F")
        subtriangulation_1 = subtriangulation_1.flatten(order="F")
        subtriangulation_2 = subtriangulation_2.flatten(order="F")

        ind = subtriangulation_0 != -1

        subtriangulation = np.concatenate((subtriangulation_0[ind, np.newaxis],
                                           subtriangulation_1[ind, np.newaxis],
                                           subtriangulation_2[ind, np.newaxis]), axis=1)

        self.triangle_to_polygon = tri_to_poly
        self.subtriangulation = subtriangulation
        self.elem_bounding_boxes = elem_bounding_boxes

        total_edge_x = total_edge_x.flatten(order="F")
        total_edge_y = total_edge_y.flatten(order="F")

        non_zeros = total_edge_x != -1

        total_edge = np.concatenate((total_edge_x[non_zeros, np.newaxis], total_edge_y[non_zeros, np.newaxis]), axis=1)

        # classify all edges

        total_edge = np.sort(total_edge, axis=1)

        sparse_mat = csr_matrix((np.tile([1], total_edge.shape[0]), (total_edge[:, 1], total_edge[:, 0])))
        i, j, s = find(sparse_mat)

        self.boundary_edges = np.concatenate((j[s == 1, np.newaxis], i[s == 1, np.newaxis]), axis=1)
        self.interior_edges = np.concatenate((j[s == 2, np.newaxis], i[s == 2, np.newaxis]), axis=1)

        edge_to_elements = {}

        for idx, element in enumerate(self.mesh.filtered_regions):
            element_edges = [(min(a, b), max(a, b)) for i, a in enumerate(element) for b in element[i + 1:]]

            for edge in element_edges:
                if edge in edge_to_elements:
                    edge_to_elements[edge].append(idx)
                else:
                    edge_to_elements[edge] = [idx]

        temp_int = [
            sorted(edge_to_elements.get((min(edge[0], edge[1]), max(edge[0], edge[1])), []))
            for edge in self.interior_edges
        ]

        self.interior_edges_to_element = np.array(temp_int)

        temp_bound = [
            edge_to_elements.get((min(edge[0], edge[1]), max(edge[0], edge[1])), [-1])[0]
            for edge in self.boundary_edges
        ]

        self.boundary_edges_to_element = np.array(temp_bound)

        # OPUNV for edges

        bd_tan_vec = self.nodes[self.boundary_edges[:, 0], :] - self.nodes[self.boundary_edges[:, 1], :]
        int_tan_vec = self.nodes[self.interior_edges[:, 0], :] - self.nodes[self.interior_edges[:, 1], :]

        bd_normalisation_consts = np.sqrt(bd_tan_vec[:, 0] ** 2 + bd_tan_vec[:, 1] ** 2)

        bd_tan_vec = np.roll(bd_tan_vec, -1, axis=1)
        bd_tan_vec[:, 1] *= -1
        bd_nor_vec = np.divide(bd_tan_vec, bd_normalisation_consts[:, np.newaxis])

        int_normalisation_consts = np.sqrt(int_tan_vec[:, 0] ** 2 + int_tan_vec[:, 1] ** 2)

        int_tan_vec = np.roll(int_tan_vec, -1, axis=1)
        int_tan_vec[:, 1] *= -1
        int_nor_vec = np.divide(int_tan_vec, int_normalisation_consts[:, np.newaxis])

        bd_outward = self.nodes[self.boundary_edges[:, 0], :] - \
            self.mesh.filtered_points[self.boundary_edges_to_element.flatten(order="F"), :]

        int_outward = self.mesh.filtered_points[self.interior_edges_to_element[:, 1], :] - \
            self.mesh.filtered_points[self.interior_edges_to_element[:, 0], :]

        bd_index = np.maximum(np.sum(bd_nor_vec * bd_outward, axis=1), 0)
        int_index = np.sum(int_nor_vec * int_outward, axis=1) < 0.0

        i = np.argwhere(bd_index == 0)
        bd_nor_vec[i, :] = -bd_nor_vec[i, :]
        int_nor_vec[int_index, :] = -int_nor_vec[int_index, :]

        self.boundary_normals = bd_nor_vec
        self.interior_normals = int_nor_vec

        # Information for the subtriangulation

        self.n_triangles = subtriangulation.shape[0]

        edge_to_triangles = {}

        for idx, triangle in enumerate(self.subtriangulation):
            edges = [
                (min(a, b), max(a, b))
                for i, a in enumerate(triangle) for b in triangle[i + 1:]
            ]

            for edge in edges:
                if edge in edge_to_triangles:
                    edge_to_triangles[edge].append(idx)
                else:
                    edge_to_triangles[edge] = [idx]

        temp_int = []

        for edge in list(self.interior_edges):
            temp_int.append(sorted(edge_to_triangles.get(tuple(edge))))

        self.interior_edges_to_triangle = np.array(temp_int)

        temp_bound = []

        for edge in self.boundary_edges:
            temp_bound.append(*edge_to_triangles.get(tuple(edge)))

        self.boundary_edges_to_triangle = np.array(temp_bound)

    def save_geometry(self, filepath: str, save_mesh: bool = False):
        with open(filepath, "wb") as file:
            pickle.dump({k: v for k, v in self.__dict__.items() if k != ('mesh' if save_mesh else '')}, file)


# Section: Testing

# from poly_mesher.poly_mesher_domain import RectangleDomain
# from poly_mesher.poly_mesher_main import poly_mesher
# from poly_mesher.poly_mesher_clean import poly_mesher_cleaner
# from poly_mesher.show_mesh import show_mesh
# import matplotlib.pyplot as plt
# from matplotlib.patches import Polygon

# np.random.seed(1337)

# domain = RectangleDomain(bounding_box=np.array([[0, 1], [0, 1]]))
# voronoi = poly_mesher(domain, max_iterations=100, n_points=10)
# pseudo_voronoi = poly_mesher_cleaner(voronoi)
#
# ouput = GeneralGeometry(pseudo_voronoi)

# show_mesh(pseudo_voronoi.vertices, pseudo_voronoi.filtered_regions, bounding_box=np.array([[0, 1], [0, 1]]))
# show_mesh(ouput.nodes, ouput.subtriangulation, bounding_box=np.array([[0, 1], [0, 1]]))
#
#
# # Test triangle to polygon
#
# fig, ax = plt.subplots()
# for k, element in enumerate(ouput.subtriangulation):
#     ax.add_patch(Polygon(ouput.nodes[element, :], linewidth=1.0, edgecolor="black"))
#     centroid = np.mean(ouput.nodes[element, :], axis=0)
#     ax.annotate(f"{ouput.triangle_to_polygon[k]}", centroid)
#
# ax.set_xlim(domain.bounding_box[0, :])
# ax.set_ylim(domain.bounding_box[1, :])
#
# plt.show()
#
# # Test boundary edges + normals
#
# fig, ax = plt.subplots()
# for k, edge in enumerate(ouput.boundary_edges):
#     ax.plot(ouput.nodes[edge, 0], ouput.nodes[edge, 1], 'o', ls='-', ms=8, c="r")
#     centroid = np.mean(ouput.nodes[edge, :], axis=0)
#     ax.quiver(*centroid, *ouput.boundary_normals[k])
#
# plt.show()
#
# # Test interior edges + normals
#
# fig, ax = plt.subplots()
# for k, edge in enumerate(ouput.interior_edges):
#     ax.plot(ouput.nodes[edge, 0], ouput.nodes[edge, 1], 'o', ls='-', ms=8, c="r")
#     centroid = np.mean(ouput.nodes[edge, :], axis=0)
#     ax.quiver(*centroid, *ouput.interior_normals[k])
#
# plt.show()
#
# # Test boundary edges to element
#
# fig, ax = plt.subplots()
#
# for k, element in enumerate(ouput.elements):
#     centroid = np.mean(ouput.nodes[element, :], axis=0)
#     ax.annotate(f"{k}", centroid)
#
# for k, edge in enumerate(ouput.boundary_edges):
#     ax.plot(ouput.nodes[edge, 0], ouput.nodes[edge, 1], 'o', ls='-', ms=8, c="r")
#     centroid = np.mean(ouput.nodes[edge, :], axis=0)
#     ax.annotate(f"{ouput.boundary_edges_to_element[k]}", centroid)
#
# ax.set_xlim(domain.bounding_box[0, :])
# ax.set_ylim(domain.bounding_box[1, :])
#
# plt.show()
#
# # Test interior edges to element
#
# fig, ax = plt.subplots()
#
# for k, element in enumerate(ouput.elements):
#     centroid = np.mean(ouput.nodes[element, :], axis=0)
#     ax.annotate(f"{k}", centroid)
#
# for k, edge in enumerate(ouput.interior_edges):
#     ax.plot(ouput.nodes[edge, 0], ouput.nodes[edge, 1], 'o', ls='-', ms=8, c="r")
#     centroid = np.mean(ouput.nodes[edge, :], axis=0)
#     ax.annotate(f"{ouput.interior_edges_to_element[k]}", centroid)
#
# ax.set_xlim(domain.bounding_box[0, :])
# ax.set_ylim(domain.bounding_box[1, :])
#
# plt.show()
#
# Test boundary edge triangle
#
# fig, ax = plt.subplots()
#
# for k, element in enumerate(ouput.subtriangulation):
#     ax.add_patch(Polygon(ouput.nodes[element, :], linewidth=1.0, edgecolor="black"))
#     centroid = np.mean(ouput.nodes[element, :], axis=0)
#     ax.annotate(f"{k}", centroid)
#
# for k, edge in enumerate(ouput.boundary_edge_triangle):
#     ax.plot(ouput.nodes[edge, 0], ouput.nodes[edge, 1], 'o', ls='-', ms=8, c="r")
#     centroid = np.mean(ouput.nodes[edge, :], axis=0)
#     ax.annotate(f"{ouput.boundary_edges_to_element_triangle[k]}", centroid)
#
# plt.show()
#
# # Test interior edge triangle
#
# fig, ax = plt.subplots()
#
# for k, element in enumerate(ouput.subtriangulation):
#     ax.add_patch(Polygon(ouput.nodes[element, :], linewidth=1.0, edgecolor="black"))
#     centroid = np.mean(ouput.nodes[element, :], axis=0)
#     ax.annotate(f"{k}", centroid)
#
# for k, edge in enumerate(ouput.interior_edge_triangle):
#     ax.plot(ouput.nodes[edge, 0], ouput.nodes[edge, 1], 'o', ls='-', ms=8, c="r")
#     centroid = np.mean(ouput.nodes[edge, :], axis=0)
#     ax.annotate(f"{ouput.interior_edges_to_element_triangle[k]}", centroid)
#
# plt.show()
#
# # Test node to triangle element
#
# fig, ax = plt.subplots()
#
# for k, element in enumerate(ouput.subtriangulation):
#     ax.add_patch(Polygon(ouput.nodes[element, :], linewidth=1.0, edgecolor="black"))
#     centroid = np.mean(ouput.nodes[element, :], axis=0)
#     ax.annotate(f"{k}", centroid)
#
#
# for k in range(ouput.nodes.shape[0]):
#     ax.plot(ouput.nodes[k, 0], ouput.nodes[k, 1], 'o', ls='-', ms=8, c="r")
#     ax.annotate(f"{ouput.node_to_triangle_element[k]}", ouput.nodes[k, :])
#
#
# ax.set_xlim(domain.bounding_box[0, :])
# ax.set_ylim(domain.bounding_box[1, :])
#
# plt.show()

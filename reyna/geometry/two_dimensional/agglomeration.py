"""
This file will contain all the code pertaining to the agglomeration set-up.
This will nee the use of metis as the base and the geometry will be build on top of this to ensure all the normals
work as expected -- As a side note I'm not sure if the outward normals really matter in the code? Need to double check
how the geometry is generated and can go from here. The set-up is simple; input the one PolyMesh object and a list of
numbers of elements required -- agglomerate from here first to the finest n_elements and then refine this again.
"""
import typing
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
from pymetis import part_graph

from scipy.sparse import csr_matrix, find

from reyna.polymesher.two_dimensional._auxilliaries.abstraction import PolyMesh

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry


@dataclass
class Agglomeration:

    def __init__(self, poly_mesh: PolyMesh, n_refinement_elements: typing.List[int]):

        self.poly_meshes: typing.List[PolyMesh] = [poly_mesh]
        self.n_refinement_elements = n_refinement_elements

        self.geometries: typing.List[DGFEMGeometry] = [DGFEMGeometry(poly_mesh)]

        self._agglomerate()

    def _agglomerate(self):

        for i, n_parts in enumerate(self.n_refinement_elements):

            adjacency_list = _adjacency_graph(self.geometries[i].interior_edges_to_element)

            membership = _metis_with_clean_up(adjacency_list, n_parts)

            agglomerated_mesh, agglomerated_geometry = _agglomeration_geometry(np.array(membership), self.geometries[i])
            self.poly_meshes.append(agglomerated_mesh)
            self.geometries.append(agglomerated_geometry)


def _adjacency_graph(interior_edges_to_element: np.ndarray) -> typing.List[np.ndarray]:
    """
    This function takes in the interior edges to edges (naturally assuming the domain is connected) and returns a list
    of the arrays with the corresponding neighbours to each element.

    Args:
        interior_edges_to_element (np.ndarray): This is the array of the elements to which each edge corresponds to.

    Returns:
        (typing.List[np.ndarray]): This is the list of neighbours to each element.
    """

    adjacency_list = [[] for _ in range(np.max(interior_edges_to_element) + 1)]

    for edge in interior_edges_to_element:
        e0, e1 = edge
        adjacency_list[e0].append(e1)
        adjacency_list[e1].append(e0)

    adjacency_list = [np.array(lst, dtype=int) for lst in adjacency_list]

    return adjacency_list


def _metis_with_clean_up(adjacency_list: typing.List[np.ndarray], n_parts: int) -> np.ndarray:

    def _connected_components_in_partition(_pid, _n):
        visited = np.zeros(_n, dtype=bool)
        _components = []

        for start in range(_n):
            # Cycle over the elements

            if membership[start] != _pid or visited[start]:
                # Check if element is not a member of the part or the element is already seen -- not interested.
                continue

            queue = deque([start])
            visited[start] = True
            _comp = [start]

            while queue:
                node = queue.popleft()
                for nbr in adjacency_list[node]:
                    # Cycle through the neighbouring elements
                    if (membership[nbr] == _pid) and not visited[nbr]:
                        # if the neighbour is in the part and hasn't been seen
                        visited[nbr] = True
                        queue.append(nbr)  # Add neighbouring element in part to queue to search further
                        _comp.append(int(nbr))

            _components.append(_comp)

        return _components

    _, membership = part_graph(n_parts, adjacency=adjacency_list)
    membership = np.array(membership, dtype=int)

    for part in range(n_parts):

        components = _connected_components_in_partition(part, len(membership))

        if len(components) <= 1:
            continue  # already connected -- move on

        components.sort(key=len, reverse=True)  # Sort by size descending

        # Reassign all other components
        for comp in components[1:]:
            # Cycle over the smaller components to connect them later
            for elem in comp:

                # Find neighboring partitions (excluding current part)
                neighbouring_parts = [membership[nbr] for nbr in adjacency_list[elem] if membership[nbr] != part]

                if neighbouring_parts:
                    # If neighbours exist -- push to connect them -- this may be a superfluous check.
                    membership[elem] = np.argmax(np.bincount(neighbouring_parts))

    return membership


def _agglomeration_geometry(membership: np.ndarray, geometry: DGFEMGeometry) -> (PolyMesh, DGFEMGeometry):
    """
    This function is a special one -- This needs to take in the metis refinement and agglomerate the geometry. This also
    returns the PolyMesh object associated with the agglomerated geometry.
    """

    elem_bounding_boxes = []
    agglomerated_areas = []
    agglomerated_filtered_regions = []

    total_edge = np.empty((0, 2), dtype=int)

    for i in range(np.max(membership) + 1):

        agglomerated_elements = np.argwhere(membership == i).ravel()  # This is linear.....I think

        if len(agglomerated_elements) == 0:
            continue

        agglomerated_areas.append(np.sum(geometry.areas[agglomerated_elements]))

        # Combine the elements here.....
        edge_count = defaultdict(int)
        for agglomerated_element in agglomerated_elements:
            edges = geometry.mesh.filtered_regions[agglomerated_element]
            _n = len(edges)
            for j in range(_n):
                v1, v2 = edges[j], edges[(j + 1) % _n]
                edge_count[tuple(sorted([v1, v2]))] += 1

        element_edges = [edge for edge, count in edge_count.items() if count == 1]  # TODO: Error \/ traced to here
        total_edge = np.concatenate((total_edge, np.array(element_edges)), axis=0)

        edge_adjacency = defaultdict(list)
        for v1, v2 in element_edges:
            edge_adjacency[v1].append(v2)
            edge_adjacency[v2].append(v1)

        # Traverse the edge to order the edges to the element
        start = min(edge_adjacency.keys())  # TODO: intermitent error here....? empty .keys()?
        element_edges = [start]
        current = start
        prev = None

        while True:
            neighbors = edge_adjacency[current]
            next_vertex = neighbors[0] if neighbors[0] != prev else neighbors[1]
            if next_vertex == start:
                break
            element_edges.append(next_vertex)
            prev, current = current, next_vertex

        # Ensure CCW
        area = 0
        for k in range(n := len(element_edges)):
            x1, y1 = geometry.nodes[element_edges[k], :]
            x2, y2 = geometry.nodes[element_edges[(k + 1) % n], :]
            area += x1 * y2 - x2 * y1
        area *= 0.5

        if area < 0:
            element_edges.reverse()

        elem_bounding_boxes.append([np.min(geometry.nodes[element_edges, 0]), np.max(geometry.nodes[element_edges, 0]),
                                    np.min(geometry.nodes[element_edges, 1]), np.max(geometry.nodes[element_edges, 1])])

        agglomerated_filtered_regions.append(np.array(element_edges))

    # Create the interior edges.
    sparse_mat = csr_matrix((np.tile([1], total_edge.shape[0]), (total_edge[:, 1], total_edge[:, 0])))
    i, j, s = find(sparse_mat)

    agglomerated_interior_edges = np.concatenate((j[s == 2, np.newaxis], i[s == 2, np.newaxis]), axis=1)

    edge_to_elements = {}

    for idx, element in enumerate(agglomerated_filtered_regions):
        element_edges = [(min(a, b), max(a, b)) for i, a in enumerate(element) for b in element[i + 1:]]

        for edge in element_edges:
            if edge in edge_to_elements:
                edge_to_elements[edge].append(idx)
            else:
                edge_to_elements[edge] = [idx]

    temp_int = [
        sorted(edge_to_elements.get((min(edge[0], edge[1]), max(edge[0], edge[1])), []))
        for edge in agglomerated_interior_edges
    ]

    interior_edges_to_element = np.array(temp_int)

    agglomerated_interior_normals = np.zeros((0, 2), dtype=float)
    if agglomerated_interior_edges.shape[0] > 1:
        # One element is not guarenteed to have any interior edges

        int_tan_vec = (geometry.nodes[agglomerated_interior_edges[:, 0], :] -
                       geometry.nodes[agglomerated_interior_edges[:, 1], :])
        int_normalisation_consts = np.sqrt(int_tan_vec[:, 0] ** 2 + int_tan_vec[:, 1] ** 2)

        int_tan_vec = np.roll(int_tan_vec, -1, axis=1)
        int_tan_vec[:, 1] *= -1
        int_nor_vec = np.divide(int_tan_vec, int_normalisation_consts[:, np.newaxis])

        # TODO: May need something similar for the advection case?
        # int_outward = (geometry.mesh.filtered_points[interior_edges_to_element[:, 1], :] -
        #                geometry.mesh.filtered_points[interior_edges_to_element[:, 0], :])
        #
        # int_index = np.sum(int_nor_vec * int_outward, axis=1) < 0.0
        # int_nor_vec[int_index, :] = -int_nor_vec[int_index, :]
        agglomerated_interior_normals = int_nor_vec

    triangle_to_agglomorated_polygon = np.array(
        [membership[geometry.triangle_to_polygon[i]] for i in range(geometry.n_triangles)]
    )

    agglomerated_poly_mesh = PolyMesh(
        vertices=geometry.nodes,  # This is never touched again -- can be empty -- left for testing
        filtered_regions=agglomerated_filtered_regions,
        filtered_points=np.empty((0, 2)),  # Never again used.
        domain=geometry.mesh.domain
    )

    agglomerated_geometry = DGFEMGeometry.__new__(DGFEMGeometry)
    agglomerated_geometry.mesh = agglomerated_poly_mesh

    # Initial information
    agglomerated_geometry.n_nodes = geometry.n_nodes
    agglomerated_geometry.n_elements = np.max(membership) + 1
    agglomerated_geometry.nodes = geometry.nodes
    agglomerated_geometry.elem_bounding_boxes = elem_bounding_boxes

    # Boundary edges and normals
    agglomerated_geometry.boundary_edges = geometry.boundary_edges
    agglomerated_geometry.boundary_edges_to_element = membership[geometry.boundary_edges_to_element]
    agglomerated_geometry.boundary_normals = geometry.boundary_normals

    # Interior edges and normals
    agglomerated_geometry.interior_edges = agglomerated_interior_edges  # TODO: optimise to reuse.
    agglomerated_geometry.interior_edges_to_element = interior_edges_to_element
    agglomerated_geometry.interior_normals = agglomerated_interior_normals

    # Subtriangulation information
    agglomerated_geometry.subtriangulation = geometry.subtriangulation
    agglomerated_geometry.n_triangles = geometry.n_triangles
    agglomerated_geometry.triangle_to_polygon = triangle_to_agglomorated_polygon

    # Final information
    agglomerated_geometry.h = None
    agglomerated_geometry.h_s = None  # These two /\ need to be reran with the bounding_boxes work and new filtered_regions
    agglomerated_geometry.areas = np.array(agglomerated_areas)

    return agglomerated_poly_mesh, agglomerated_geometry

import typing
from dataclasses import dataclass
from collections import defaultdict, deque

import numpy as np
from pymetis import part_graph
from shapely import Polygon, Point

from reyna.polymesher.two_dimensional._auxilliaries.abstraction import PolyMesh

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry


@dataclass
class Agglomeration:

    def __init__(self, poly_mesh: PolyMesh, n_refinement_elements: typing.List[int]):

        if np.any(n_refinement_elements != np.unique(n_refinement_elements)[::-1]):
            raise ValueError("Please make sure the elements in 'n_refinement_elements' are unique and in reverse "
                             "order.")

        self.n_refinement_elements = n_refinement_elements

        self.poly_meshes: typing.List[PolyMesh] = [poly_mesh]
        self.geometries: typing.List[DGFEMGeometry] = [DGFEMGeometry(poly_mesh)]

        self._agglomerate()

    def _agglomerate(self) -> None:

        """
        The main function which performs the agglomeration steps. This recursively generates the meshes in such a
        way that they are nested. The corresponding DGFEMGeometry objects are generated simultaneously in an optimised
        manner.

        Returns:
            None

        """

        for i, n_parts in enumerate(self.n_refinement_elements):

            adjacency_list = _adjacency_graph(self.geometries[i].interior_edges_to_element)
            membership = _metis_with_clean_up(adjacency_list, n_parts)

            agglomerated_mesh, agglomerated_geometry = _agglomeration_geometry(membership, self.geometries[i])
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

    """
    This function takes in an adjacency list and a given number of elements and performs a METIS step to partition the
    graph into 'n_parts' (or there abouts). METIS is not flawless and can produce disconnected subgraphs; this function
    is designed to clean this up. This function retains the largest subsection and splits the remaining pieces into
    their most common neighbours (by number of shared facets). Additionally, METIS may not be able to produce exactly
    'n_parts' subgraphs; in this case, it tends to produce a few less elements than expected.

    Args:
        adjacency_list (typing.List[np.ndarray]): This is the list of neighbours to each element.
        n_parts (int): The number of parts to partition the graph into.

    Returns:
        (np.ndarray): This is an array containing the 'membership' of each element to its agglomerated subgraph.

    """

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
            continue  # already connected (or empty) -- move on

        components.sort(key=len, reverse=True)  # Sort by size descending

        # Reassign all other components
        for comp in components[1:]:
            # Cycle over the smaller components to connect them with other parts
            for elem in comp:

                # Find neighboring partitions (excluding current part)
                neighbouring_parts = [membership[nbr] for nbr in adjacency_list[elem] if membership[nbr] != part]

                if neighbouring_parts:
                    # If neighbours exist -- push to connect them -- this may be a superfluous check.
                    membership[elem] = np.argmax(np.bincount(neighbouring_parts))

    # Tidy up duplicated elements.
    _, membership = np.unique(membership, return_inverse=True)

    return membership


def _agglomeration_geometry(membership: np.ndarray, geometry: DGFEMGeometry) -> (PolyMesh, DGFEMGeometry):
    """
    This function is a special one -- This needs to take in the metis refinement and agglomerate the geometry. This also
    returns the PolyMesh object associated with the agglomerated geometry.

    Args:
        membership (np.ndarray): This is an array containing the 'membership' of each element to its agglomerated
            subgraph.
        geometry (DGFEMGeometry): This is the DGFEM geometry object for the original geometry; the agglomerated geometry
            will take on variables associated to this.

    Returns:
        (PolyMesh, DGFEMGeometry): The agglomerated PolyMesh object and its associated geometry.

    """

    elem_bounding_boxes = []
    agglomerated_areas = []
    agglomerated_filtered_regions = []
    h_s = []

    for i in range(np.max(membership)+1):

        # Get the elements associated with the new region
        agglomerated_elements = np.argwhere(membership == i).ravel()
        agglomerated_areas.append(np.sum(geometry.areas[agglomerated_elements]))  # areas = sum(sub_areas)

        # Combine the elements here.....
        edge_count = defaultdict(int)
        for agglomerated_element in agglomerated_elements:
            edges = geometry.mesh.filtered_regions[agglomerated_element]
            _n = len(edges)
            for j in range(_n):
                v1, v2 = edges[j], edges[(j + 1) % _n]
                edge_count[tuple(sorted([v1, v2]))] += 1

        element_edges = [edge for edge, count in edge_count.items() if count == 1]

        # Agglomerated adjacency for the vertices
        edge_adjacency = defaultdict(list)
        for v1, v2 in element_edges:
            edge_adjacency[v1].append(v2)
            edge_adjacency[v2].append(v1)

        # Traverse the edge to order the edges to the element
        start = min(edge_adjacency.keys())
        element_edges: typing.List[int] = [start]
        current = start
        prev = None

        while True:
            neighbors = edge_adjacency[current]
            if neighbors:
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
        area *= 0.5  # This matches the sum formula above (up to sign error)

        if area < 0:
            element_edges.reverse()

        elem_bounding_boxes.append([np.min(geometry.nodes[element_edges, 0]), np.max(geometry.nodes[element_edges, 0]),
                                    np.min(geometry.nodes[element_edges, 1]), np.max(geometry.nodes[element_edges, 1])])

        poly = Polygon(geometry.nodes[element_edges, :])
        box = poly.minimum_rotated_rectangle
        _x, _y = box.exterior.coords.xy
        edge_length = (Point(_x[0], _y[0]).distance(Point(_x[1], _y[1])),
                       Point(_x[1], _y[1]).distance(Point(_x[2], _y[2])))
        h_s.append(max(edge_length))

        agglomerated_filtered_regions.append(np.array(element_edges))

    # 'initialise' the geometry object
    agglomerated_geometry = DGFEMGeometry.__new__(DGFEMGeometry)

    # Initial information
    agglomerated_geometry.h_s = np.array(h_s)
    agglomerated_geometry.h = np.max(agglomerated_geometry.h_s)
    agglomerated_geometry.areas = np.array(agglomerated_areas)
    agglomerated_geometry.elem_bounding_boxes = elem_bounding_boxes

    # Agglomeratred poly-mesh information for both the mesh and the geometry.
    agglomerated_poly_mesh = PolyMesh(
        vertices=geometry.nodes,  # This is never touched again -- can be empty -- left for testing
        filtered_regions=agglomerated_filtered_regions,
        filtered_points=np.empty((0, 2)),  # Never again used.
        domain=geometry.mesh.domain
    )

    agglomerated_geometry.mesh = agglomerated_poly_mesh

    edge_to_elements = defaultdict(list)

    # Build global edge -> elements map
    for elem, vertices in enumerate(agglomerated_filtered_regions):
        n = len(vertices)
        for i in range(n):
            vertices: typing.List[float]  # This is superfluous
            edge = (min(vertices[i], vertices[(i + 1) % n]), max(vertices[i], vertices[(i + 1) % n]))
            edge_to_elements[edge].append(elem)

    agglomerated_interior_edges = []
    interior_edges_to_element = []

    # Loop over edges to detect interior edges
    for edge, elems in edge_to_elements.items():
        if len(elems) == 2:
            agglomerated_interior_edges.append(edge)
            interior_edges_to_element.append(elems)

    agglomerated_interior_edges = np.array(agglomerated_interior_edges, dtype=int)
    interior_edges_to_element = np.array(interior_edges_to_element, dtype=int)

    agglomerated_interior_normals = np.zeros((0, 2), dtype=float)

    if agglomerated_interior_edges.shape[0] > 1:
        # A One element domain is not guarenteed to have any interior edges

        int_tan_vec = (geometry.nodes[agglomerated_interior_edges[:, 0], :] -
                       geometry.nodes[agglomerated_interior_edges[:, 1], :])
        int_normalisation_consts = np.sqrt(int_tan_vec[:, 0] ** 2 + int_tan_vec[:, 1] ** 2)

        int_tan_vec = np.roll(int_tan_vec, -1, axis=1)
        int_tan_vec[:, 1] *= -1
        int_nor_vec = np.divide(int_tan_vec, int_normalisation_consts[:, np.newaxis])

        agglomerated_interior_normals = int_nor_vec

    # Interior edges and normals
    agglomerated_geometry.interior_edges = agglomerated_interior_edges
    agglomerated_geometry.interior_edges_to_element = interior_edges_to_element
    agglomerated_geometry.interior_normals = agglomerated_interior_normals

    # Boundary edges and normals
    agglomerated_geometry.boundary_edges = geometry.boundary_edges
    agglomerated_geometry.boundary_edges_to_element = membership[geometry.boundary_edges_to_element]
    agglomerated_geometry.boundary_normals = geometry.boundary_normals

    # Subtriangulation information
    agglomerated_geometry.subtriangulation = geometry.subtriangulation
    agglomerated_geometry.n_triangles = geometry.n_triangles
    agglomerated_geometry.triangle_to_polygon = membership[geometry.triangle_to_polygon]

    # Further information
    agglomerated_geometry.n_nodes = geometry.n_nodes
    agglomerated_geometry.n_elements = membership.max() + 1
    agglomerated_geometry.nodes = geometry.nodes

    return agglomerated_poly_mesh, agglomerated_geometry

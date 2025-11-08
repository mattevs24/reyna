import numpy as np

from reyna.polymesher.two_dimensional.domains import RectangleDomain, CircleDomain, CircleCircleDomain, HornDomain
from reyna.polymesher.two_dimensional.main import poly_mesher

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry


class TestGeometry:

    def test_number_of_nodes_and_elements(self):
        """ Confirms the number of elements and points are equal between the geometry and mesh objects. """

        dom = CircleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=32)

        geometry = DGFEMGeometry(mesh)

        assert geometry.n_elements == len(mesh.filtered_regions), 'Must be 32 elements'
        assert geometry.n_nodes == mesh.vertices.shape[0], 'Mesh vertices and geometry nodes must be equal shape.'

    def test_subtriangulation(self):
        """ Tests whether the subtriangulation is a true subtriangulation and whether the indexing between triangles
        and polygons is correct. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=32)

        geometry = DGFEMGeometry(mesh)

        for i, triangle in enumerate(geometry.subtriangulation):

            triangle_nodes = geometry.nodes[triangle]
            element_nodes = geometry.nodes[geometry.mesh.filtered_regions[geometry.triangle_to_polygon[i]]]

            contains = np.any(np.all(triangle_nodes[:, None] == element_nodes, axis=2), axis=0)

            assert sum(contains) == 3, 'Subtriangle must be contained in the element.'

    def test_bounding_boxes_bound(self):
        """ Tests whether the bounding box bounds are in fact bounding the elements. """
        dom = CircleCircleDomain()
        mesh = poly_mesher(dom, n_points=128)

        geometry = DGFEMGeometry(mesh)

        epsilon = 1e-9

        for i, element in enumerate(geometry.mesh.filtered_regions):

            element_nodes = geometry.nodes[element, :]
            bounding_box = geometry.elem_bounding_boxes[i]

            assert (bounding_box[0] - epsilon) <= np.min(element_nodes[:, 0]) <= (bounding_box[1] + epsilon), \
                f'Element not contained in bounding box.'
            assert (bounding_box[0] - epsilon) <= np.max(element_nodes[:, 0]) <= (bounding_box[1] + epsilon), \
                f'Element not contained in bounding box.'
            assert (bounding_box[2] - epsilon) <= np.min(element_nodes[:, 1]) <= (bounding_box[3] + epsilon), \
                f'Element not contained in bounding box.'
            assert (bounding_box[2] - epsilon) <= np.max(element_nodes[:, 1]) <= (bounding_box[3] + epsilon), \
                f'Element not contained in bounding box.'

    def test_boundary_edges(self):
        """ Tests whether the boundary edges are on the boundary -- I.e. checks the indexing"""

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=32)

        geometry = DGFEMGeometry(mesh)

        epsilon = 1e-9

        for b_edge in geometry.boundary_edges:
            edge_nodes = geometry.nodes[b_edge]

            assert ((np.all(np.abs(edge_nodes[:, 1]) < epsilon) or np.all(np.abs(edge_nodes[:, 1] - 1.0) < epsilon)) or
                    (np.all(np.abs(edge_nodes[:, 0]) < epsilon) or np.all(np.abs(edge_nodes[:, 0] - 1.0) < epsilon))), \
                'Boundary edge not contained in the boundary'

    def test_interior_edges(self):
        """ Tests whether the interior edges are on the boundary -- checks the indexing"""

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=32)

        geometry = DGFEMGeometry(mesh)

        epsilon = 1e-9

        for edge in geometry.interior_edges:
            edge_nodes = geometry.nodes[edge]

            assert (np.all(-epsilon <= np.min(edge_nodes, axis=0)) and
                    np.all(np.min(edge_nodes, axis=0) <= 1.0 + epsilon) and
                    np.all(-epsilon <= np.max(edge_nodes, axis=0)) and
                    np.all(np.max(edge_nodes, axis=0) <= 1.0 + epsilon)), \
                'Interior edge not contained in the interior'

    def test_boundary_edges_to_element(self):
        """ Tests whether the boundary edges to element indexing is correct. """

        dom = HornDomain()
        mesh = poly_mesher(dom, n_points=64)

        geometry = DGFEMGeometry(mesh)

        for i, edge in enumerate(geometry.boundary_edges):

            element_idx = geometry.boundary_edges_to_element[i]

            edge_nodes = geometry.nodes[edge]
            element_nodes = geometry.nodes[geometry.mesh.filtered_regions[element_idx]]

            contains = np.any(np.all(edge_nodes[:, None] == element_nodes, axis=2), axis=0)

            assert sum(contains) == 2, 'Boundary edge must be contained in the corresponding element.'

    def test_interior_edges_to_element(self):
        """ Tests whether the interior edges to element indexing is correct. """

        dom = HornDomain()
        mesh = poly_mesher(dom, n_points=64)

        geometry = DGFEMGeometry(mesh)

        for i, edge in enumerate(geometry.interior_edges):

            element_idxs = geometry.interior_edges_to_element[i]

            edge_nodes = geometry.nodes[edge]
            element_nodes_0 = geometry.nodes[geometry.mesh.filtered_regions[element_idxs[0]]]
            element_nodes_1 = geometry.nodes[geometry.mesh.filtered_regions[element_idxs[0]]]

            contains_0 = np.any(np.all(edge_nodes[:, None] == element_nodes_0, axis=2), axis=0)
            contains_1 = np.any(np.all(edge_nodes[:, None] == element_nodes_1, axis=2), axis=0)

            assert sum(contains_0) == 2, 'Interior edge must be contained in one corresponding element.'
            assert sum(contains_1) == 2, 'Interior edge must be contained in the other corresponding element.'

    def test_boundary_normals(self):
        """ Tests whether the boundary normals are perpendicular to the boudnary edges. """

        dom = CircleCircleDomain()
        mesh = poly_mesher(dom, n_points=64)

        geometry = DGFEMGeometry(mesh)
        epsilon = 1e-9

        for i, edge in enumerate(geometry.boundary_edges):

            normal = geometry.boundary_normals[i]
            edge_vector = geometry.nodes[edge[1]] - geometry.nodes[edge[1]]

            assert np.linalg.norm(normal * edge_vector) < epsilon, \
                'Boundary normal must be perpendicular to the corresponding boundary edge.'

    def test_interior_normals(self):
        """Tests whether the interior normals are perpendicular to the interior edges. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=64)

        geometry = DGFEMGeometry(mesh)
        epsilon = 1e-9

        for i, edge in enumerate(geometry.interior_edges):
            normal = geometry.interior_normals[i]
            edge_vector = geometry.nodes[edge[1]] - geometry.nodes[edge[1]]

            assert np.linalg.norm(normal * edge_vector) < epsilon, \
                'Interior normal must be perpendicular to the corresponding interior edge.'

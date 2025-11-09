import pytest
import numpy as np

from reyna.polymesher.two_dimensional.domains import RectangleDomain, CircleDomain, CircleCircleDomain, HornDomain
from reyna.polymesher.two_dimensional.main import poly_mesher


class TestPolyMesher:

    def test_number_of_elements_random(self):
        """ Tests the number of elements for a random point set. """
        n_elements = 32
        dom = CircleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=n_elements)

        assert len(mesh.filtered_regions) == 32, 'Mesh does not contain 32 elements as expected'

    def test_number_of_elements_grid(self):
        """ Tests the number of elements for a grid point set. """
        n = 8
        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_xy=(n, n))

        assert len(mesh.filtered_regions) == 64, 'Mesh does not contain 64 elements as expected'

    def test_number_of_elements_failure(self):
        """ Tests the failure of no initial point set. """
        dom = HornDomain()

        with pytest.raises(AttributeError):
            _ = poly_mesher(dom)

    def test_fixed_points(self):
        """ Tests fixed points are indeed fixed. """
        fixed_points = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
        dom = RectangleDomain(np.array([[0, 1], [0, 1]]), fixed_points=fixed_points)
        mesh = poly_mesher(dom, n_points=50)

        assert np.array_equal(mesh.filtered_points[: fixed_points.shape[0]], fixed_points), 'Fixed points do not match'

    def test_elements_contained_in_domain(self):
        """ Tests whether all elements lie within the domain. """
        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=100)

        epsilon = 1e-9

        for element in mesh.filtered_regions:
            element_vertices = mesh.vertices[element]
            assert -epsilon <= np.min(element_vertices) <= 1.0 + epsilon, 'Element is not contained in the domain.'
            assert -epsilon <= np.max(element_vertices) <= 1.0 + epsilon, 'Element is not contained in the domain.'

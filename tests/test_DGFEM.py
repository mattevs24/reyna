import pytest
import numpy as np

from reyna.polymesher.two_dimensional.domains import RectangleDomain, CircleDomain, RectangleCircleDomain, HornDomain
from reyna.polymesher.two_dimensional.main import poly_mesher

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry

from reyna.DGFEM.two_dimensional.main import DGFEM


np.random.seed(1142)


class TestDGFEM:

    # Parameter tests to begin

    def test_no_advection_or_diffusion(self):
        """Test the failure when neither advection nor diffusion are used. """

        dom = CircleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=32)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        with pytest.raises(ValueError):
            dg.add_data(dirichlet_bcs=lambda x: np.zeros(x.shape[0]))

    def test_dirichlet_bcs_input(self):
        """ Test is the bcs input is functional as expected when no boundary conditions are inputted. """

        dom = CircleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=32)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        advection = lambda x: np.ones(x.shape, dtype=float)

        with pytest.raises(ValueError):
            dg.add_data(advection=advection)

    def test_diffusion_polynomial_degree_warning(self):
        """ Test to see if the warning and correction is made when the polynomial degree is set too low. """

        dom = RectangleCircleDomain()
        mesh = poly_mesher(dom, n_points=128)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry, polynomial_degree=0)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 1.0
                out[i, 1, 1] = 1.0
            return out

        with pytest.warns(UserWarning, match='The polynomial degree given is too low'):
            dg.add_data(diffusion=diffusion, dirichlet_bcs=lambda x: np.zeros(x.shape[0]))

    def test_verbose_setting(self):
        """ Test to see if the verbose counter fails as expected. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=16)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 1.0
                out[i, 1, 1] = 1.0
            return out

        dg.add_data(diffusion=diffusion, dirichlet_bcs=lambda x: np.zeros(x.shape[0]))

        with pytest.raises(AssertionError):
            dg.dgfem(verbose=2)

    def test_advection_constant_solution_polynomial_degree_one(self):
        """ Test to see if constant solutions are captured exactly for advection equations with p=1 polynomials. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=16)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        advection = lambda x: np.ones(x.shape, dtype=float)

        dg.add_data(
            advection=advection,
            dirichlet_bcs=lambda x: np.ones(x.shape[0], dtype=float)
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(exact_solution=lambda x: np.ones(x.shape[0], dtype=float))

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_advection_linear_solution_polynomial_degree_one(self):
        """ Test to see if linear solutions are captured exactly for advection equations with p=1 polynomials. """

        dom = HornDomain()
        mesh = poly_mesher(dom, n_points=128)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        advection = lambda x: np.ones(x.shape, dtype=float)
        forcing = lambda x: 2.0 * np.ones(x.shape[0], dtype=float)

        dg.add_data(
            advection=advection,
            forcing=forcing,
            dirichlet_bcs=lambda x: x[:, 0] + x[:, 1]
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(exact_solution=lambda x: x[:, 0] + x[:, 1])

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_advection_quadratic_solution_polynomial_degree_two(self):
        """ Test to see if quadratic solutions are captured exactly for advection equations with p=2 polynomials. """

        dom = RectangleCircleDomain()
        mesh = poly_mesher(dom, n_points=128)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry, polynomial_degree=2)

        advection = lambda x: np.ones(x.shape, dtype=float)
        forcing = lambda x: 2.0 * (x[:, 0] + x[:, 1])

        dg.add_data(
            advection=advection,
            forcing=forcing,
            dirichlet_bcs=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(exact_solution=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2)

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_advection_reaction_constant_solution_polynomial_degree_one(self):
        """ Test to see if constant solutions are captured exactly for a-r equations with p=1 polynomials. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=16)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        advection = lambda x: np.ones(x.shape, dtype=float)
        reaction = lambda x: np.ones(x.shape[0], dtype=float)
        forcing = lambda x: np.ones(x.shape[0], dtype=float)

        dg.add_data(
            advection=advection,
            reaction=reaction,
            forcing=forcing,
            dirichlet_bcs=lambda x: np.ones(x.shape[0], dtype=float)
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(exact_solution=lambda x: np.ones(x.shape[0], dtype=float))

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_advection_reaction_linear_solution_polynomial_degree_one(self):
        """ Test to see if linear solutions are captured exactly for a-r equations with p=1 polynomials. """

        dom = HornDomain()
        mesh = poly_mesher(dom, n_points=256)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        advection = lambda x: np.ones(x.shape, dtype=float)
        reaction = lambda x: 4.0 * np.ones(x.shape[0], dtype=float)
        forcing = lambda x: 2.0 * np.ones(x.shape[0], dtype=float) + 4.0 * (x[:, 0] + x[:, 1])

        dg.add_data(
            advection=advection,
            reaction=reaction,
            forcing=forcing,
            dirichlet_bcs=lambda x: x[:, 0] + x[:, 1]
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(exact_solution=lambda x: x[:, 0] + x[:, 1])

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_advection_reaction_quadratic_solution_polynomial_degree_two(self):
        """ Test to see if quadratic solutions are captured exactly for a-r equations with p=2 polynomials. """

        dom = RectangleCircleDomain()
        mesh = poly_mesher(dom, n_points=128)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry, polynomial_degree=2)

        advection = lambda x: np.ones(x.shape, dtype=float)
        reaction = lambda x: 0.5 * np.ones(x.shape[0], dtype=float)
        forcing = lambda x: 2.0 * (x[:, 0] + x[:, 1]) + 0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2)

        dg.add_data(
            advection=advection,
            reaction=reaction,
            forcing=forcing,
            dirichlet_bcs=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(exact_solution=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2)

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_diffusion_missing_errors(self):
        """ Test to see if missing grad_u_exact behaves as expected in diffusion error checking. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=16)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 1.0
                out[i, 1, 1] = 1.0
            return out

        dg.add_data(
            diffusion=diffusion,
            dirichlet_bcs=lambda x: np.ones(x.shape[0], dtype=float)
        )
        dg.dgfem(solve=True)

        with pytest.raises(ValueError):
            l2_norm, dg_norm, _ = dg.errors(exact_solution=lambda x: np.ones(x.shape[0], dtype=float))

    def test_diffusion_constant_solution_polynomial_degree_one(self):
        """ Test to see if constant solutions are captured exactly for diffusion equations with p=1 polynomials. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=16)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 1.0
                out[i, 1, 1] = 1.0
            return out

        dg.add_data(
            diffusion=diffusion,
            dirichlet_bcs=lambda x: np.ones(x.shape[0], dtype=float)
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(
            exact_solution=lambda x: np.ones(x.shape[0], dtype=float),
            grad_exact_solution=lambda x: np.zeros(x.shape, dtype=float)
        )

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_diffusion_linear_solution_polynomial_degree_one(self):
        """ Test to see if linear solutions are captured exactly for diffusion equations with p=1 polynomials. """

        dom = CircleDomain()
        mesh = poly_mesher(dom, n_points=256)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 1.0
                out[i, 1, 1] = 1.0
            return out

        dg.add_data(
            diffusion=diffusion,
            dirichlet_bcs=lambda x: x[:, 0] + x[:, 1]
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(
            exact_solution=lambda x: x[:, 0] + x[:, 1],
            grad_exact_solution=lambda x: np.ones(x.shape, dtype=float)
        )

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_diffusion_quadratic_solution_polynomial_degree_two(self):
        """ Test to see if quadratic solutions are captured exactly for diffusion equations with p=2 polynomials. """

        dom = RectangleCircleDomain()
        mesh = poly_mesher(dom, n_points=128)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry, polynomial_degree=2)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 2.0
                out[i, 1, 1] = 2.0
            return out

        forcing = lambda x: -8.0 * np.ones(x.shape[0], dtype=float)

        dg.add_data(
            diffusion=diffusion,
            forcing=forcing,
            dirichlet_bcs=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(
            exact_solution=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
            grad_exact_solution=lambda x: 2.0 * np.concatenate((x[:, 0][:, None], x[:, 1][:, None]), axis=1)
        )

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_non_equal_diffusion_quadratic_solution_polynomial_degree_two(self):
        """ Test to see if quadratuc solutions are captured exactly for anisotropic diffusion equations with p=2
        polynomials. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=64)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry, polynomial_degree=2)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 2.5
                out[i, 1, 1] = 0.5
            return out

        forcing = lambda x: -6.0 * np.ones(x.shape[0], dtype=float)

        dg.add_data(
            diffusion=diffusion,
            forcing=forcing,
            dirichlet_bcs=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(
            exact_solution=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
            grad_exact_solution=lambda x: 2.0 * np.concatenate((x[:, 0][:, None], x[:, 1][:, None]), axis=1)
        )

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_non_diagonal_diffusion_quadratic_solution_polynomial_degree_two_1(self):
        """ Test to see if quadratic solutions are captured exactly for non-diagonal diffusion equations with p=2
        polynomials. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=64)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry, polynomial_degree=2)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 2.0
                out[i, 0, 1] = 1.0
                out[i, 1, 0] = 1.0
                out[i, 1, 1] = 2.0
            return out

        forcing = lambda x: -8.0 * np.ones(x.shape[0], dtype=float)

        dg.add_data(
            diffusion=diffusion,
            forcing=forcing,
            dirichlet_bcs=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(
            exact_solution=lambda x: x[:, 0] ** 2 + x[:, 1] ** 2,
            grad_exact_solution=lambda x: 2.0 * np.concatenate((x[:, 0][:, None], x[:, 1][:, None]), axis=1)
        )

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_non_diagonal_diffusion_quadratic_solution_polynomial_degree_two_2(self):
        """ Test to see if quadratic solutions are captured exactly for non-diagonal diffusion equations with p=2
        polynomials. """

        dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
        mesh = poly_mesher(dom, n_points=64)

        geometry = DGFEMGeometry(mesh)
        dg = DGFEM(geometry, polynomial_degree=2)

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 2.0
                out[i, 0, 1] = 1.0
                out[i, 1, 0] = 1.0
                out[i, 1, 1] = 2.0
            return out

        forcing = lambda x: -2.0 * np.ones(x.shape[0], dtype=float)

        dg.add_data(
            diffusion=diffusion,
            forcing=forcing,
            dirichlet_bcs=lambda x: x[:, 0] * x[:, 1]
        )
        dg.dgfem(solve=True)

        l2_norm, dg_norm, _ = dg.errors(
            exact_solution=lambda x: x[:, 0] * x[:, 1],
            grad_exact_solution=lambda x: np.concatenate((x[:, 1][:, None], x[:, 0][:, None]), axis=1)
        )

        epsilon = 1e-10

        assert l2_norm < epsilon, 'L2 error is expected to be near zero.'
        assert dg_norm < epsilon, 'dG error is expected to be near zero.'

    def test_benchmark_diffusion(self):
        """ Benchmark diffusion equations with p=1 polynomials. """
        np.random.seed(1142)

        n_elements = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        h_s = []
        dg_norms = []
        l2_norms = []
        h1_norms = []

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 1.0
                out[i, 1, 1] = 1.0
            return out

        reaction = lambda x: np.pi ** 2 * np.ones(x.shape[0], dtype=float)
        forcing = lambda x: 3.0 * np.pi ** 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

        bcs = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

        solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

        def grad_solution(x: np.ndarray):
            u_x = np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
            u_y = np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

            return np.vstack((u_x, u_y)).T

        domain = RectangleDomain(np.array([[0, 1], [0, 1]]))

        for n_r in n_elements:

            poly_mesh = poly_mesher(domain, max_iterations=10, n_points=n_r)
            geometry = DGFEMGeometry(poly_mesh)

            dg = DGFEM(geometry, polynomial_degree=1)
            dg.add_data(
                diffusion=diffusion,
                reaction=reaction,
                dirichlet_bcs=bcs,
                forcing=forcing
            )

            dg.dgfem(solve=True)

            l2_error, dg_error, h1_error = dg.errors(
                exact_solution=solution,
                grad_exact_solution=grad_solution,
            )

            h_s.append(geometry.h)
            l2_norms.append(l2_error)
            dg_norms.append(dg_error)
            h1_norms.append(h1_error)

        l2_powers = np.diff(np.log(l2_norms)) / np.diff(np.log(h_s))
        dg_powers = np.diff(np.log(dg_norms)) / np.diff(np.log(h_s))
        h1_powers = np.diff(np.log(h1_norms)) / np.diff(np.log(h_s))

        assert np.exp(np.mean(np.log(l2_powers[-3:]))) > 1.6, 'L2 error decay is not behaving as expected.'
        assert np.exp(np.mean(np.log(dg_powers[-3:]))) > 0.7, 'dG error decay is not behaving as expected.'
        assert np.exp(np.mean(np.log(h1_powers[-3:]))) > 0.7, 'H1 error decay is not behaving as expected.'

    def test_benchmark_advection_reaction(self):
        """ Benchmark advection equations with p=1 polynomials. """
        np.random.seed(1337)

        n_elements = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        h_s = []
        dg_norms = []
        l2_norms = []

        advection = lambda x: np.ones(x.shape, dtype=float)
        reaction = lambda x: np.pi * np.ones(x.shape[0], dtype=float)
        forcing = lambda x: (np.pi * (np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
                                      np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])) +
                             np.pi * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]))

        bcs = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

        solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

        domain = RectangleDomain(bounding_box=np.array([[0, 1], [0, 1]]))

        for n_r in n_elements:

            poly_mesh = poly_mesher(domain, max_iterations=10, n_points=n_r)
            geometry = DGFEMGeometry(poly_mesh)

            dg = DGFEM(geometry, polynomial_degree=1)
            dg.add_data(
                advection=advection,
                reaction=reaction,
                dirichlet_bcs=bcs,
                forcing=forcing
            )
            dg.dgfem(solve=True)

            l2_error, dg_error, _ = dg.errors(
                exact_solution=solution,
                div_advection=lambda x: np.zeros(x.shape[0]),
            )

            h_s.append(geometry.h)
            l2_norms.append(l2_error)
            dg_norms.append(dg_error)

        l2_powers = np.diff(np.log(l2_norms)) / np.diff(np.log(h_s))
        dg_powers = np.diff(np.log(dg_norms)) / np.diff(np.log(h_s))

        assert np.exp(np.mean(np.log(l2_powers[-3:]))) > 1.6, 'L2 error decay is not behaving as expected.'
        assert np.exp(np.mean(np.log(dg_powers[-3:]))) > 1.2, 'dG error decay is not behaving as expected.'

    def test_benchmark_advection_diffusion_reaction(self):
        """ Benchmark the full advection-diffusion-reaction equations with p=3 polynomials and a non-diagonal diffusion
        tensor. """
        np.random.seed(1142)

        n_elements = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

        domain = RectangleDomain(np.array([[0, 1], [0, 1]]))

        h_s = []
        dg_norms = []
        l2_norms = []
        h1_norms = []

        solution = lambda x: 0.5 * np.exp(1.0 - x[:, 0]) * np.exp(x[:, 1] - 1)

        def grad_solution(x: np.ndarray):
            u_x = -0.5 * np.exp(1.0 - x[:, 0]) * np.exp(x[:, 1] - 1)
            u_y = 0.5 * np.exp(1.0 - x[:, 0]) * np.exp(x[:, 1] - 1)

            return np.vstack((u_x, u_y)).T

        def diffusion(x):
            out = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
            for i in range(x.shape[0]):
                out[i, 0, 0] = 2.0
                out[i, 1, 0] = 1.0
                out[i, 0, 1] = 1.0
                out[i, 1, 1] = 2.0
            return out

        def advection(x: np.ndarray):
            u_x = 2.0 * np.ones(x.shape[0], dtype=float)
            u_y = np.ones(x.shape[0], dtype=float)

            return np.vstack((u_x, u_y)).T

        reaction = lambda x: 2 * np.ones(x.shape[0], dtype=float)
        forcing = lambda x: -0.5 * np.exp(1.0 - x[:, 0]) * np.exp(x[:, 1] - 1)

        for n_r in n_elements:
            poly_mesh = poly_mesher(domain, max_iterations=10, n_points=n_r)
            geometry = DGFEMGeometry(poly_mesh)

            dg = DGFEM(geometry, polynomial_degree=3)
            dg.add_data(
                diffusion=diffusion,
                advection=advection,
                reaction=reaction,
                dirichlet_bcs=solution,
                forcing=forcing
            )

            dg.dgfem(solve=True)

            l2_error, dg_error, h1_error = dg.errors(
                exact_solution=solution,
                grad_exact_solution=grad_solution,
            )

            h_s.append(geometry.h)
            l2_norms.append(l2_error)
            dg_norms.append(dg_error)
            h1_norms.append(h1_error)

        l2_powers = np.diff(np.log(l2_norms)) / np.diff(np.log(h_s))
        dg_powers = np.diff(np.log(dg_norms)) / np.diff(np.log(h_s))
        h1_powers = np.diff(np.log(h1_norms)) / np.diff(np.log(h_s))

        assert np.exp(np.mean(np.log(l2_powers[-3:]))) > 3.6, 'L2 error decay is not behaving as expected.'
        assert np.exp(np.mean(np.log(dg_powers[-3:]))) > 2.7, 'dG error decay is not behaving as expected.'
        assert np.exp(np.mean(np.log(h1_powers[-3:]))) > 2.7, 'H1 error decay is not behaving as expected.'

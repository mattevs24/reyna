import typing
import time

import numpy as np
from numba import njit, f8
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry

from reyna.DGFEM.two_dimensional._auxilliaries.boundary_information import BoundaryInformation
from reyna.DGFEM.two_dimensional._auxilliaries.polygonal_basis_utils import Basis_index2D

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.full_assembly import localstiff, int_localstiff

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.boundary_assembly import (local_advection_inflow,
                                                                                  local_diffusion_dirichlet)

from reyna.DGFEM.two_dimensional._auxilliaries.error_assembly.boundary_assembly import error_d_face, error_a_face
from reyna.DGFEM.two_dimensional._auxilliaries.error_assembly.full_assembly import error_element, error_interface

from reyna.DGFEM.two_dimensional.plotter import plot_DG


class DGFEM:
    """
    This class comtains all the methods required for the DGFEM solver.

    Attributes:
        geometry (DGFEMGeometry): The geometry associated with the computation domain for the problem.
        h (float): The maximal cell diameter of the mesh.
        polydegree (int): The highest total degree polynomial space required.

        advection (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The advection coeffient.
        diffusion (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The diffusion coeffient.
        reaction (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The reaction coeffient.
        forcing (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The forcing term.
        dirichlet_bcs (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The dirichlet boundary conditions.

        boundary_information (typing.Optional[BoundaryInformation]): The information associated with the boundary of the
        computational domain and PDE coefficients.
        sigma_D (float): The global penalisation coefficient.

        orders (typing.Optional[np.ndarray]): The orders of the tensor-Legendre polynomials.

        dim_elem (typing.Optional[int]): The local finite element space dimension.
        dim_system (typing.Optional[int]): The global system dimension.

        solution (typing.Optional[np.ndarray]): The solution vector for the coefficients of the Legendre polynomials.

        B (typing.Optional[csr_matrix]): The global stiffness matrix (populated when .dgfem() method is called).
        L (typing.Optional[np.ndarray]): The global forcng vector (populated when .dgfem() method is called).

        element_reference_quadrature (typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]): The 2d reference
        quadrature rule.
        edge_reference_quadrature (typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]): The 1d reference quadrature
        rule.

    Methods:
        add_data(...): Add the function coefficients for the non-negative characteristic second order PDE.
        dgfem(...): Run the solver with the information and geometry inputted.
        errors(...): Given the numerically solved PDE, find the errors associated with the numerical solution.
        plot_DG(...): Plot the DGFEM solution.

    See Also:
        - The reference and example notebooks for examples on how this class is used fully.
    """

    def __init__(self, geometry: DGFEMGeometry, polynomial_degree: int = 1):
        """
        Initializes DGFEM with required base parameters.

        Args:
            geometry (DGFEMGeometry): The geometry associated with the computation domain for the problem.
            polynomial_degree (int): The highest total degree polynomial space required.

        """

        self.geometry: DGFEMGeometry = geometry
        self.h: float = geometry.h
        self.polydegree: int = polynomial_degree

        # Problem Functions
        self.advection = None
        self.diffusion = None
        self.reaction = None
        self.forcing = None
        self.dirichlet_bcs = None

        self.boundary_information: typing.Optional[BoundaryInformation] = None
        self.sigma_D = 10.0

        # Method Parameters
        self.orders = None

        self.dim_elem: typing.Optional[int] = None
        self.dim_system: typing.Optional[int] = None

        self.solution: typing.Optional[np.ndarray] = None

        # Stiffness matrix + Data vector

        self.B: typing.Optional[csr_matrix] = None
        self.L: typing.Optional[np.ndarray] = None

        # Initialise quadrature parameters

        self.element_reference_quadrature: typing.Optional[typing.Tuple[np.ndarray, np.ndarray]] = None
        self.edge_reference_quadrature: typing.Optional[typing.Tuple[np.ndarray, np.ndarray]] = None
        self._intialise_quadrature()

    def add_data(self,
                 advection: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
                 diffusion: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
                 reaction: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
                 forcing: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
                 dirichlet_bcs: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None):

        """
        In this method, we add the data to the problem at hand. We define the relevant advection, diffusion, reaction
        and source terms as well as the corresponding boundary conditions. Defaults to 'None', which simplifies the
        later numerical method.

        This method also generates additional information about the boundary conditions relating to what boundary
        conditions need to be applied and where.

        Args:
            advection: The advection component of the PDE. Must be able to handle inputs with dimension (N,2) and output
            arrays with dimension (N,2). I.e. the N advection coefficients at the given N 2D input locations.
            diffusion: The diffusion component of the PDE. Must be able to handle inputs with dimension (N,2) and output
            arrays with dimension (N,2,2). I.e the N diffusion tensors at the given N 2D input locations. We note that
            the diffusion tensor must have non-negative characteristic form.
            reaction: The reaction component of the PDE. Must be able to handle inputs with dimension (N,2) and output
            arrays with dimension (N,). I.e. the N reaction coefficients at the given N 2D input locations.
            forcing: The forcing component of the PDE. Must be able to handle inputs with dimension (N,2) and output
            arrays with dimension (N,). I.e. the N forcing values at the given N 2D input locations.
            dirichlet_bcs: The Dirichlet boundary conditions associated with the PDE. Must be able to handle inputs
            with dimension (N,2) and output arrays with dimension (N,). I.e. the N boundary values at the given N 2D
            input locations. This can be mixed with relavent Neumann boundary conditions or stand-alone.

        Raises:
            ValueError: If Dirichlet boundary conditions are not present.
            ValueError: If no advection or diffusion terms are present.

        Notes:
            - It's important to note the shapes of the functions that are inputted. All need to be able ot take in a
              (N,2) array and return an array of varying shapes depending on the dimension of the output of the
              coefficient.
            - These functions must also be 'numba' compatible. This means certain 'numpy' functions are not allowed.

        See Also:
            - [numba compatible functions](https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html)

        """

        self.advection = njit(f8[:, :](f8[:, :]))(advection) if advection is not None else None
        self.diffusion = njit(f8[:, :, :](f8[:, :]))(diffusion) if diffusion is not None else None
        self.reaction = njit(f8[:](f8[:, :]))(reaction) if reaction is not None else None
        self.forcing = njit(f8[:](f8[:, :]))(forcing) if forcing is not None else None
        self.dirichlet_bcs = njit(f8[:](f8[:, :]))(dirichlet_bcs) if dirichlet_bcs is not None else None

        if self.advection is None and self.diffusion is None:
            raise ValueError('No advection or diffusion specified.')

        if self.dirichlet_bcs is None:
            raise ValueError('Must have Dirichlet boundary conditions.')

        self.boundary_information = self._define_boundary_information()

    def dgfem(self, solve: bool = True, verbose: int = 0):

        """
        This is the main method to the class and generates all the stiffness matrices and data vector
        values. It also generates the solution vector to the problem.

        Args:
            solve (bool): This method generates the linear system associated with the DGFEM solution to the inputted
            problem. Selecting solve will also solve this linear system.
            verbose (int) : This is the verbose level of the method. 0 is no verbose, 1 gives the assembly time.

        Returns:
            None

        Raises:
            ValueError: If the parameter 'verbose' is not contained in the right interval.

        """

        assert 0 <= verbose <= 1, ValueError('"verbose" must be either 0 or 1.')

        # Generate the basic information for the method, shared by all the methods.
        self.orders = Basis_index2D(self.polydegree)  # Legendre polynomial orders

        self.dim_elem: int = np.shape(self.orders)[0]
        self.dim_system: int = self.dim_elem * self.geometry.n_elements

        _time = time.time()

        self.B, self.L = self._stiffness_matrix()
        self.B += self._interior_stiffness_matrix()

        if self.diffusion is not None:

            diff_B, diff_f = self._diffusion_bcs_contribution()

            self.B -= diff_B
            self.L -= diff_f

        if self.advection is not None:

            adv_B, adv_f = self._advection_bcs_contribution()

            self.B -= adv_B
            self.L -= adv_f

        if verbose == 1:
            print(f"Assembly: {time.time() - _time}")

        if solve:
            self.solution = spsolve(self.B, self.L)

    def _intialise_quadrature(self) -> None:
        """
        This function generates the quadrature rule required for the scheme.

        Returns:
            None
        """

        quadrature_order = self.polydegree + 1

        x, w_x = leggauss(quadrature_order)
        y, w_y = roots_jacobi(quadrature_order, 1, 0)

        quad_x = np.reshape(np.repeat(x, w_y.shape[0]), (-1, 1))
        quad_y = np.reshape(np.tile(y, w_x.shape[0]), (-1, 1), order='F')
        weights = (w_x[:, None] * w_y).flatten().reshape(-1, 1)

        # Duffy transform to the reference triangle points
        shiftpoints = np.hstack((0.5 * (1.0 + quad_x) * (1.0 - quad_y) - 1.0, quad_y))
        ref_points = 0.5 * shiftpoints + 0.5

        # TODO: run through where these functions are used and fix the awkward [:, None] parts on them to avoid this

        self.element_reference_quadrature = (weights, ref_points)
        self.edge_reference_quadrature = (w_x[:, None], x[:, None])

    def _define_boundary_information(self, **kwargs) -> BoundaryInformation:
        """
        This method splits the boundary into the component pieces to which the Dirichlet boundary conditions are
        applied. This generates the elliptic Dirichlet boundary as well as the hyperbolic inflow and outflow
        information. For now this code assumes that the Hausdorff measure of the Neumann boundary is 0 and hence we may
        only consider the elliptical dirichlet portion of the boundary

        Returns:
            BoundaryInformation: An object containing all the relavent boundary information.
        """

        boundary_information = BoundaryInformation(self.geometry, **kwargs)
        boundary_information.split_boundaries(self.advection, self.diffusion)

        return boundary_information

    def _stiffness_matrix(self) -> (csr_matrix, csr_matrix):
        """
        This method generates the local stiffness matrix for upwind/IP scheme in use as well as the loading/forcing
        vector.

        Returns:
            (csr_matrix, csr_matrix): A sparse matrix and vector containing the contributions of the diffusion,
            advection, reaction and forcing terms to the stiffness matrix and loading vector.
        """

        i = np.zeros((self.dim_elem ** 2, self.geometry.n_elements), dtype=int)
        j = np.zeros((self.dim_elem ** 2, self.geometry.n_elements), dtype=int)
        s = np.zeros((self.dim_elem ** 2, self.geometry.n_elements))

        s_f = np.zeros(self.dim_system)

        ind_x: np.ndarray
        ind_y: np.ndarray

        ind_x, ind_y = np.meshgrid(np.arange(self.dim_elem), np.arange(self.dim_elem))
        ind_x, ind_y = ind_x.flatten('F'), ind_y.flatten('F')

        for t in range(self.geometry.n_elements):
            i[:, t] = self.dim_elem * t + ind_x
            j[:, t] = self.dim_elem * t + ind_y

        for t in range(self.geometry.n_triangles):
            local_triangle = self.geometry.subtriangulation[t, :]
            element_idx = self.geometry.triangle_to_polygon[t]

            local_stiff, local_forcing = localstiff(
                self.geometry.nodes[local_triangle, :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.element_reference_quadrature,
                self.orders,
                self.diffusion,
                self.advection,
                self.reaction,
                self.forcing
            )
            s[:, element_idx] += local_stiff.flatten('F')
            if local_forcing is not None:
                s_f[element_idx * self.dim_elem: (element_idx + 1) * self.dim_elem] += local_forcing

        stiffness_matrix = csr_matrix((s.flatten('F'), (i.flatten('F'), j.flatten('F'))),
                                      shape=(self.dim_system, self.dim_system))

        forcing_contribution_vector = csr_matrix(
            (s_f, (np.arange(self.dim_system), np.zeros(self.dim_system, dtype=int))),
            shape=(self.dim_system, 1)
        )

        return stiffness_matrix, forcing_contribution_vector

    def _interior_stiffness_matrix(self) -> csr_matrix:
        """
        This method generates the interior stiffness matrix for upwind/IP scheme in use.

        Returns:
            csr_matrix: A sparse matrix containing both diffusion and advection contributions over the edges of the
            computational mesh.
        """

        i = np.zeros((4 * self.dim_elem ** 2, self.geometry.interior_edges.shape[0]), dtype=int)
        j = np.zeros((4 * self.dim_elem ** 2, self.geometry.interior_edges.shape[0]), dtype=int)
        s = np.zeros((4 * self.dim_elem ** 2, self.geometry.interior_edges.shape[0]))

        for t in range(self.geometry.interior_edges.shape[0]):
            elem_oneface = self.geometry.interior_edges_to_element[t, :]

            # Direction of normal is predetermined! -- Points from elem_oneface[0] to elem_oneface[1]
            interface = int_localstiff(
                self.geometry.nodes[self.geometry.interior_edges[t, :], :],
                self.geometry.elem_bounding_boxes[elem_oneface[0]],
                self.geometry.elem_bounding_boxes[elem_oneface[1]],
                self.edge_reference_quadrature,
                self.orders,
                self.geometry.nodes[self.geometry.mesh.filtered_regions[elem_oneface[0]], :],
                self.geometry.nodes[self.geometry.mesh.filtered_regions[elem_oneface[1]], :],
                self.geometry.areas[elem_oneface[0]],
                self.geometry.areas[elem_oneface[1]],
                self.polydegree,
                self.sigma_D,
                self.geometry.interior_normals[t, :],
                self.diffusion,
                self.advection
            )

            ind1 = np.arange(elem_oneface[0] * self.dim_elem, (elem_oneface[0] + 1) * self.dim_elem)
            ind2 = np.arange(elem_oneface[1] * self.dim_elem, (elem_oneface[1] + 1) * self.dim_elem)
            ind = np.concatenate((ind1, ind2))

            i[:, t] = np.repeat(ind, 2 * self.dim_elem)
            j[:, t] = np.tile(ind, 2 * self.dim_elem)
            s[:, t] = interface.flatten('F')

        int_ip_matrix = csr_matrix((s.flatten('F'), (i.flatten('F'), j.flatten('F'))),
                                   shape=(self.dim_system, self.dim_system))

        return int_ip_matrix

    def _diffusion_bcs_contribution(self) -> (csr_matrix, csr_matrix):
        """
        This method generates the portion of the stiffness matrix and vector associated with the elliptic
        boundary for the IP scheme used for the diffusion term.

        Returns:
            (csr_matrix, csr_matrix): An sparse matrix and a sparse vector containing the contribution of the diffusion
            term to the elliptic boundary for the striffness matrix and forcing vectors.
        """
        i = np.zeros((self.dim_elem ** 2, self.geometry.n_elements), dtype=int)
        j = np.zeros((self.dim_elem ** 2, self.geometry.n_elements), dtype=int)
        s = np.zeros((self.dim_elem ** 2, self.geometry.n_elements))

        s_f = np.zeros((self.dim_system, 1))

        ind_x: np.ndarray
        ind_y: np.ndarray

        ind_x, ind_y = np.meshgrid(np.arange(self.dim_elem), np.arange(self.dim_elem))
        ind_x, ind_y = ind_x.flatten('F'), ind_y.flatten('F')

        for t in range(self.boundary_information.elliptical_indecies.shape[0]):
            element_idx = self.geometry.boundary_edges_to_element[t]
            i[:, element_idx] = self.dim_elem * element_idx + ind_x
            j[:, element_idx] = self.dim_elem * element_idx + ind_y

        for v in list(self.boundary_information.elliptical_indecies):
            element_idx = self.geometry.boundary_edges_to_element[v]

            local_bc_diff, local_bcs_forcing = local_diffusion_dirichlet(
                self.geometry.nodes[self.geometry.boundary_edges[v, :], :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.edge_reference_quadrature,
                self.orders,
                self.geometry.nodes[self.geometry.mesh.filtered_regions[element_idx], :],
                self.geometry.areas[element_idx],
                self.polydegree,
                self.sigma_D,
                self.geometry.boundary_normals[v, :],
                self.dirichlet_bcs,
                self.diffusion
            )

            s[:, element_idx] += local_bc_diff.flatten('F')
            s_f[element_idx * self.dim_elem: (element_idx + 1) * self.dim_elem] += local_bcs_forcing

        diffusion_bcs_stiffness_matrix = csr_matrix((s.flatten('F'), (i.flatten('F'), j.flatten('F'))),
                                                    shape=(self.dim_system, self.dim_system))

        forcing_bcs_diffusion = csr_matrix(s_f)

        return diffusion_bcs_stiffness_matrix, forcing_bcs_diffusion

    def _advection_bcs_contribution(self) -> (csr_matrix, csr_matrix):
        """
        This method generates the portion of the stiffness matrix and vector associated with the inflow
        boundary for the upwind scheme used.

        Returns:
            (csr_matrix, csr_matrix): An sparse matrix and a sparse vector containing the contribution of the advection
            term to the inflow boundary for the striffness matrix and forcing vectors.
        """

        i = np.zeros((self.dim_elem ** 2, len(self.boundary_information.inflow_indecies)), dtype=int)
        j = np.zeros((self.dim_elem ** 2, len(self.boundary_information.inflow_indecies)), dtype=int)
        s = np.zeros((self.dim_elem ** 2, len(self.boundary_information.inflow_indecies)))

        s_f = np.zeros((self.dim_system, 1))

        ind_x: np.ndarray
        ind_y: np.ndarray

        ind_x, ind_y = np.meshgrid(np.arange(self.dim_elem), np.arange(self.dim_elem))
        ind_x, ind_y = ind_x.flatten('F'), ind_y.flatten('F')

        for t, v in enumerate(list(self.boundary_information.inflow_indecies)):
            element_idx = self.geometry.boundary_edges_to_element[v]

            local_bc_adv, forcing_bcs_adv = local_advection_inflow(
                self.geometry.nodes[self.geometry.boundary_edges[v, :], :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.edge_reference_quadrature,
                self.orders,
                self.geometry.boundary_normals[v, :],
                self.advection,
                self.dirichlet_bcs
            )

            i[:, t] = self.dim_elem * element_idx + ind_x
            j[:, t] = self.dim_elem * element_idx + ind_y
            s[:, t] += local_bc_adv.flatten('F')

            s_f[element_idx * self.dim_elem: (element_idx + 1) * self.dim_elem] += forcing_bcs_adv

        advection_bcs_stiffness_matrix = csr_matrix((s.flatten('F'), (i.flatten('F'), j.flatten('F'))),
                                                    shape=(self.dim_system, self.dim_system))
        forcing_inflow_bcs_vector = csr_matrix(s_f)

        return advection_bcs_stiffness_matrix, forcing_inflow_bcs_vector

    def errors(self,
               exact_solution: typing.Callable[[np.ndarray], np.ndarray],
               grad_exact_solution: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
               div_advection: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None
               ) -> typing.Tuple[float, float, typing.Optional[float]]:

        """
        This function calculates three (semi-) norms associated with the discontinuous Galerkin scheme employed
        here. First the L2 norm as well as the DG norm (the sum of the convective and diffusive norms) and the H1
        semi-norm.

        Args:
            exact_solution: The exact solution. Must take in an array of size (N,2) and return an array which
            outputs values in a (N,) array.
            grad_exact_solution: The gradient of the exact solution. Must take in an array of size (N,2) and
            return an array which outputs values in a (N,2) array.
            div_advection: The divergence of the advection coeffiecient. This is required for the DG norm.
            This has to take in an array of size (N,2) and return an array which outputs values in a (N,) array.

        Returns:
            (float, float, typing.Optional[float], typing.Optional[dict]): The L2 norm, the DG norm,
            the H1 semi-norm and the dg subnorm dict respectively.

        Raises:
              ValueError: If the 'solution' property is not filled before running.
              ValueError: If 'self.diffusion' and the parameter 'grad_exact_solution' are not both fulled or empty.

        """

        if self.solution is None:
            raise ValueError('Need to run the .dgfem() method to generate a solution before calculating an error.')

        if (self.diffusion is None) != (grad_exact_solution is None):
            raise ValueError('Need to input both or neither "diffusion" and "grad_u_exact".')

        # Input validation/cleaning
        if self.reaction is None:
            self.reaction = lambda x: np.zeros(x.shape[0])
        elif self.advection is None:
            div_advection = lambda x: np.zeros(x.shape[0])

        auxilliary_function = lambda x: self.reaction(x) - 0.5 * div_advection(x)

        l2_error, dg_error, h1_error = 0.0, 0.0, 0.0

        for t in range(self.geometry.n_triangles):
            local_triangle = self.geometry.subtriangulation[t, :]
            element_idx = self.geometry.triangle_to_polygon[t]

            l2_subnorm, dg_subnorm, h1_subnorm = error_element(
                self.geometry.nodes[local_triangle, :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.element_reference_quadrature,
                self.orders,
                self.solution[element_idx * self.dim_elem:(element_idx + 1) * self.dim_elem],
                exact_solution,
                grad_exact_solution,
                self.diffusion,
                auxilliary_function
            )

            l2_error += l2_subnorm
            dg_error += dg_subnorm

            if self.diffusion is not None:
                h1_error += h1_subnorm

        for t in range(self.geometry.interior_edges.shape[0]):
            elem_oneface = self.geometry.interior_edges_to_element[t, :]

            dg_subnorm = error_interface(
                self.geometry.nodes[self.geometry.interior_edges[t, :], :],
                self.geometry.elem_bounding_boxes[elem_oneface[0]],
                self.geometry.elem_bounding_boxes[elem_oneface[1]],
                self.edge_reference_quadrature,
                self.orders,
                self.geometry.nodes[self.geometry.mesh.filtered_regions[elem_oneface[0]], :],
                self.geometry.nodes[self.geometry.mesh.filtered_regions[elem_oneface[1]], :],
                self.geometry.areas[elem_oneface[0]],
                self.geometry.areas[elem_oneface[1]],
                self.polydegree,
                self.sigma_D,
                self.solution[elem_oneface[0] * self.dim_elem:(elem_oneface[0] + 1) * self.dim_elem],
                self.solution[elem_oneface[1] * self.dim_elem:(elem_oneface[1] + 1) * self.dim_elem],
                self.geometry.interior_normals[t, :],
                self.diffusion,
                self.advection
            )

            dg_error += dg_subnorm

        if self.diffusion is not None:

            for v in list(self.boundary_information.elliptical_indecies):
                element_idx = self.geometry.boundary_edges_to_element[v]

                dg_subnorm = error_d_face(
                    self.geometry.nodes[self.geometry.boundary_edges[v, :], :],
                    self.geometry.elem_bounding_boxes[element_idx],
                    self.edge_reference_quadrature,
                    self.orders,
                    self.geometry.nodes[self.geometry.mesh.filtered_regions[element_idx], :],
                    self.geometry.areas[element_idx],
                    self.polydegree,
                    self.sigma_D,
                    self.solution[element_idx * self.dim_elem:(element_idx + 1) * self.dim_elem],
                    self.geometry.boundary_normals[v, :],
                    exact_solution,
                    self.diffusion
                )

                dg_error += dg_subnorm

        if self.advection is not None:

            for t in range(self.geometry.boundary_edges.shape[0]):
                elem_bdface = self.geometry.boundary_edges_to_element[t]

                dg_subnorm = error_a_face(
                    self.geometry.nodes[self.geometry.boundary_edges[t, :], :],
                    self.geometry.elem_bounding_boxes[elem_bdface],
                    self.edge_reference_quadrature,
                    self.orders,
                    self.solution[elem_bdface * self.dim_elem:(elem_bdface + 1) * self.dim_elem],
                    self.geometry.boundary_normals[t, :],
                    exact_solution,
                    self.advection
                )

                dg_error += dg_subnorm

        l2_error = np.sqrt(l2_error)
        h1_error = np.sqrt(h1_error)
        dg_error = np.sqrt(dg_error)

        if self.diffusion is None:
            return l2_error, dg_error, None

        return l2_error, dg_error, h1_error

    def plot_DG(self):
        """
        This function plots the DGFEM solution using Matplotlib. For more customisability, use the 'plot_DG' function in
        './plotter'. This is a quick reference plotter.

        Returns:
            None

        Raises:
            ValueError: If there is not a populated 'solution' property, this function returns an error.

        Notes:
            - This function is only able to plot the solutions which are piecewise linear (i.e. polynomial degree 1)

        """

        if self.solution is None:
            raise ValueError("Need to provide a solution using either the method '.dgfem' or an external function.")

        plot_DG(self.solution, self.geometry, self.polydegree)

import typing
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry

from reyna.DGFEM.two_dimensional._auxilliaries.boundary_information import BoundaryInformation
from reyna.DGFEM.two_dimensional._auxilliaries.polygonal_basis_utils import Basis_index2D

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.diffusion_assembly import localstiff_diffusion_bcs
from reyna.DGFEM.two_dimensional._auxilliaries.assembly.advection_assembly import localinflowface
from reyna.DGFEM.two_dimensional._auxilliaries.assembly.forcing_assembly import C_vecDiriface, vect_inflowDiriface
from reyna.DGFEM.two_dimensional._auxilliaries.assembly.full_assembly import localstiff, int_localstiff
from reyna.DGFEM.two_dimensional._auxilliaries.assembly.assembly_aux import quad_GL, quad_GJ1

from reyna.DGFEM.two_dimensional._auxilliaries.error_assembly.boundary_assembly import error_bd_face, error_cr_bd_face
from reyna.DGFEM.two_dimensional._auxilliaries.error_assembly.full_assembly import error_element, error_interface


class DGFEM:

    def __init__(self, geometry: DGFEMGeometry, polynomial_degree: int = 0):
        """
        Class initialisation

        In this (default) method, we define all the necessary objects for the numerical method itself. This requires
        just the geometry object which defines all useful features of the mesh as well as the degree of polynomial
        approximation required.
        """
        self.geometry = geometry
        self.h = geometry.h
        self.polydegree = polynomial_degree

        # Problem Functions
        self.advection = None
        self.diffusion = None
        self.reaction = None
        self.forcing = None
        self.dirichlet_bcs = None

        self.boundary_information: typing.Optional[BoundaryInformation] = None
        self.sigma_D = 10 * (self.polydegree + 1) * (self.polydegree + 2)

        # Method Parameters

        self.fekete_integration_degree: int = 2 * polynomial_degree + 1
        self.polynomial_indecies = None

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
        """

        self.advection = advection
        self.diffusion = diffusion
        self.reaction = reaction
        self.forcing = forcing
        self.dirichlet_bcs = dirichlet_bcs

        if self.dirichlet_bcs is None:
            raise ValueError('Must have Dirichlet boundary conditions.')

        self.boundary_information = self._define_boundary_information()

    def dgfem(self, solve=True):

        """
        This is the main method to the class and generates all the stiffness matrices and data vector
        values. It also generates the solution vector to the problem.
        """

        # Generate the basic information for the method, shared by all the methods.
        self.polynomial_indecies = Basis_index2D(self.polydegree)

        self.dim_elem = np.shape(self.polynomial_indecies)[0]
        self.dim_system = self.dim_elem * self.geometry.n_elements

        _time = time.time()

        self.B, self.L = self._stiffness_matrix()
        self.B += self._interior_stiffness_matrix()

        if self.diffusion is not None:

            self.B -= self._diffusion_boundary_conditions()
            self.L -= self._forcing_boundary_conditions()

        if self.advection is not None:

            self.B -= self._advection_contribution_inflow_boundary()
            self.L -= self._advection_bcs_vector()

        print(f"Assembly: {time.time() - _time}")

        if solve:
            self.solution = spsolve(self.B, self.L)

    def _intialise_quadrature(self):

        quadrature_order = int(np.ceil(0.5 * (self.fekete_integration_degree + 1)))
        w_x, x = quad_GL(quadrature_order)
        w_y, y = quad_GJ1(quadrature_order)

        quad_x = np.reshape(np.repeat(x, w_y.shape[0]), (-1, 1))
        quad_y = np.reshape(np.tile(y, w_x.shape[0]), (-1, 1), order='F')
        weights = (w_x[:, None] * w_y).flatten().reshape(-1, 1)

        # The duffy points and the reference triangle points.
        shiftpoints = np.hstack((0.5 * (1.0 + quad_x) * (1.0 - quad_y) - 1.0, quad_y))
        ref_points = 0.5 * shiftpoints + 0.5

        self.element_reference_quadrature = (weights, ref_points)
        self.edge_reference_quadrature = (w_x, x)

    def _define_boundary_information(self, **kwargs) -> BoundaryInformation:
        """
        This method splits the boundary into the component pieces to which the Dirichlet and Neumann boundary conditions
        are applied. This generates both the elliptic Dirishlet and Neumann boundary as well as the hyperbolic inflow
        and outflow information.

        Returns:
            BoundaryInformation: An object containing all the relavent information.
        """

        # For now this code assumes that the Hausdorff measure of the Neumann boundary is 0 and hence
        # we may only consider the elliptical dirichlet portion of the boundary

        boundary_information = BoundaryInformation(**kwargs)
        boundary_information.split_boundaries(self.geometry, self.advection, self.diffusion)
        # boundary_information.plot_boundaries(self.geometry)

        return boundary_information

    def _stiffness_matrix(self):
        """
        Assemble the local stiffness matrices and load vector for the 2D integral components.
        """

        i = np.zeros((self.dim_elem ** 2, self.geometry.n_elements), dtype=int)
        j = np.zeros((self.dim_elem ** 2, self.geometry.n_elements), dtype=int)
        s = np.zeros((self.dim_elem ** 2, self.geometry.n_elements))

        s_f = np.zeros(self.dim_system)

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
                self.polynomial_indecies,
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

    def _interior_stiffness_matrix(self):

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
                self.polynomial_indecies,
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

    def _diffusion_boundary_conditions(self):

        i = np.zeros((self.dim_elem ** 2, self.geometry.n_elements), dtype=int)
        j = np.zeros((self.dim_elem ** 2, self.geometry.n_elements), dtype=int)
        s = np.zeros((self.dim_elem ** 2, self.geometry.n_elements))

        ind_x, ind_y = np.meshgrid(np.arange(self.dim_elem), np.arange(self.dim_elem))
        ind_x, ind_y = ind_x.flatten('F'), ind_y.flatten('F')

        for t in range(self.boundary_information.elliptical_indecies.shape[0]):
            element_idx = self.geometry.boundary_edges_to_element[t]
            i[:, element_idx] = self.dim_elem * element_idx + ind_x
            j[:, element_idx] = self.dim_elem * element_idx + ind_y

        for v in list(self.boundary_information.elliptical_indecies):
            element_idx = self.geometry.boundary_edges_to_element[v]

            local_bc_diff = localstiff_diffusion_bcs(
                self.geometry.nodes[self.geometry.boundary_edges[v, :], :],
                self.geometry.boundary_normals[v, :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.edge_reference_quadrature,
                self.polynomial_indecies,
                self.geometry.nodes[self.geometry.mesh.filtered_regions[element_idx], :],
                self.geometry.areas[element_idx],
                self.polydegree,
                self.sigma_D,
                self.diffusion
            )

            s[:, element_idx] += local_bc_diff.flatten('F')

        diffusion_bcs_stiffness_matrix = csr_matrix((s.flatten('F'), (i.flatten('F'), j.flatten('F'))),
                                                    shape=(self.dim_system, self.dim_system))

        return diffusion_bcs_stiffness_matrix

    def _forcing_boundary_conditions(self):

        i = np.arange(self.dim_system)
        s = np.zeros(self.dim_system)

        for v in list(self.boundary_information.elliptical_indecies):
            element_idx = self.geometry.boundary_edges_to_element[v]

            vec_DiriBDface = C_vecDiriface(
                self.geometry.nodes[self.geometry.boundary_edges[v, :], :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.geometry.boundary_normals[v, :],
                self.edge_reference_quadrature,
                self.polynomial_indecies,
                self.geometry.nodes[self.geometry.mesh.filtered_regions[element_idx], :],
                self.geometry.areas[element_idx],
                self.polydegree,
                self.sigma_D,
                self.dirichlet_bcs,
                self.diffusion
            )

            s[element_idx * self.dim_elem: (element_idx + 1) * self.dim_elem] += vec_DiriBDface

        forcing_boundary_conditions_matrix = csr_matrix((s, (i, np.zeros(self.dim_system, dtype=int))),
                                                        shape=(self.dim_system, 1))

        return forcing_boundary_conditions_matrix

    def _advection_contribution_inflow_boundary(self):

        """
        This method generates the portion of the stiffness matrix associated with the inflow
        boundary for the upwind scheme used.
        """

        i = np.zeros((self.dim_elem ** 2, len(self.boundary_information.inflow_indecies)), dtype=int)
        j = np.zeros((self.dim_elem ** 2, len(self.boundary_information.inflow_indecies)), dtype=int)
        s = np.zeros((self.dim_elem ** 2, len(self.boundary_information.inflow_indecies)))

        ind_x, ind_y = np.meshgrid(np.arange(self.dim_elem), np.arange(self.dim_elem))
        ind_x, ind_y = ind_x.flatten('F'), ind_y.flatten('F')

        for t, v in enumerate(list(self.boundary_information.inflow_indecies)):
            element_idx = self.geometry.boundary_edges_to_element[v]

            local_bc_adv = localinflowface(
                self.geometry.nodes[self.geometry.boundary_edges[v, :], :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.geometry.boundary_normals[v, :],
                self.edge_reference_quadrature,
                self.polynomial_indecies,
                self.advection
            )

            i[:, t] = self.dim_elem * element_idx + ind_x
            j[:, t] = self.dim_elem * element_idx + ind_y
            s[:, t] += local_bc_adv.flatten('F')

        diffusion_bcs_stiffness_matrix = csr_matrix((s.flatten('F'), (i.flatten('F'), j.flatten('F'))),
                                                    shape=(self.dim_system, self.dim_system))

        return diffusion_bcs_stiffness_matrix

    def _advection_bcs_vector(self):

        """
        This method generates the contribution of the advection field to the data vector.
        """

        i = np.arange(self.dim_system)
        s = np.zeros(self.dim_system)

        for v in list(self.boundary_information.inflow_indecies):
            element_idx = self.geometry.boundary_edges_to_element[v]

            vec_DiriBDface = vect_inflowDiriface(
                self.geometry.nodes[self.geometry.boundary_edges[v, :], :],
                self.geometry.boundary_normals[v, :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.edge_reference_quadrature,
                self.polynomial_indecies,
                self.advection,
                self.dirichlet_bcs
            )

            s[element_idx * self.dim_elem: (element_idx + 1) * self.dim_elem] += vec_DiriBDface.flatten()

        forcing_inflow_bcs_vector = csr_matrix((s, (i, np.zeros(self.dim_system, dtype=int))),
                                               shape=(self.dim_system, 1))

        return forcing_inflow_bcs_vector

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
            (float, float, typing.Optional[float]): The L2 norm, the DG norm and the H1 semi-norm respectively.
         """

        if self.solution is None:
            raise ValueError('Need to run the .dgfem() method to generate a solution before calculating an error.')

        if self.reaction is None:
            self.reaction = lambda x: np.zeros(x.shape[0])
        elif self.advection is None:
            div_advection = lambda x: np.zeros(x.shape[0])

        self.auxilliary_function = lambda x: self.reaction(x) + 0.5 * div_advection(x)

        l2_error, dg_error, h1_error = 0.0, 0.0, 0.0

        for t in range(self.geometry.n_triangles):
            local_triangle = self.geometry.subtriangulation[t, :]
            element_idx = self.geometry.triangle_to_polygon[t]

            l2_subnorm, dg_subnorm, h1_subnorm = error_element(
                self.geometry.nodes[local_triangle, :],
                self.geometry.elem_bounding_boxes[element_idx],
                self.element_reference_quadrature,
                self.polynomial_indecies,
                self.solution[element_idx * self.dim_elem:(element_idx + 1) * self.dim_elem],
                exact_solution,
                self.diffusion,
                grad_exact_solution,
                self.auxilliary_function
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
                self.polynomial_indecies,
                self.geometry.nodes[self.geometry.mesh.filtered_regions[elem_oneface[0]], :],
                self.geometry.nodes[self.geometry.mesh.filtered_regions[elem_oneface[1]], :],
                self.geometry.areas[elem_oneface[0]],
                self.geometry.areas[elem_oneface[1]],
                self.polydegree,
                self.sigma_D,
                self.geometry.interior_normals[t, :],
                self.solution[elem_oneface[0] * self.dim_elem:(elem_oneface[0] + 1) * self.dim_elem],
                self.solution[elem_oneface[1] * self.dim_elem:(elem_oneface[1] + 1) * self.dim_elem],
                self.diffusion,
                self.advection
            )

            dg_error += dg_subnorm

        if self.diffusion is not None:

            for v in list(self.boundary_information.elliptical_indecies):
                element_idx = self.geometry.boundary_edges_to_element[v]

                dg_subnorm = error_bd_face(
                    self.geometry.nodes[self.geometry.boundary_edges[v, :], :],
                    self.geometry.elem_bounding_boxes[element_idx],
                    self.solution[element_idx * self.dim_elem:(element_idx + 1) * self.dim_elem],
                    self.geometry.boundary_normals[v, :],
                    self.edge_reference_quadrature,
                    self.polynomial_indecies,
                    exact_solution,
                    self.diffusion,
                    self.geometry.nodes[self.geometry.mesh.filtered_regions[element_idx], :],
                    self.geometry.areas[element_idx],
                    self.polydegree,
                    self.sigma_D
                )

                dg_error += dg_subnorm

        if self.advection is not None:
            # error_cr_bd_face

            for t in range(self.geometry.boundary_edges.shape[0]):
                elem_bdface = self.geometry.boundary_edges_to_element[t]

                dg_subnorm = error_cr_bd_face(
                    self.geometry.nodes[self.geometry.boundary_edges[t, :], :],
                    self.geometry.elem_bounding_boxes[elem_bdface],
                    self.solution[elem_bdface * self.dim_elem:(elem_bdface + 1) * self.dim_elem],
                    self.geometry.boundary_normals[t, :],
                    self.edge_reference_quadrature,
                    self.polynomial_indecies,
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

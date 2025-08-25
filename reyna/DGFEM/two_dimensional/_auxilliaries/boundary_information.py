import typing

import numpy as np
import matplotlib.pyplot as plt

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry


class BoundaryInformation:
    """
    This object defines the boundary information for a PDE problem. This class takes in the mesh geometry as well as the
    information from the corresponding PDE and generates the boundary information from them.

    Attributes:
        geometry (DGFEMGeometry): The mesh geometry associated with the computational domain.

        tolerence (float): The tolerence for the decision on the boundary information. Decides whether the edge is
        inflow/outflow or elliptic.

        elliptical_indecies (typing.Optional[np.ndarray]): The array of indecies of the elliptical boundary edges. These
        are the indecies of the .boundary_edges property of the Geometry object.
        inflow_indecies (typing.Optional[np.ndarray]): The array of indecies of the inflow boundary edges. These
        are the indecies of the .boundary_edges property of the Geometry object.

        generated_boundary_information (bool): A bool indicating whether the boundary information is generated or not.

    Methods:
        split_boundaries(...): Generates boundary information from mesh geometry and PDE coeffcients.
        plot_boundaries(): Plots the boundary information generated from the PDE coefficients.
    """

    def __init__(self, geometry: DGFEMGeometry, **kwargs):
        """Initializes ClassName with the given parameters.

        Args:
            geometry (DGFEMGeometry): The geometry object associated with a given computational domain.
            **kwargs : Additional parameters to tune the boudnary information. 'tolerence' is the only current keyword.

        Raises:
            ValueError: If any parameter is invalid.
        """

        self.geometry = geometry

        self.tolerence = 1e-12
        if 'tolerence' in kwargs:
            self.tolerence = kwargs.pop('tolerence')

        self.elliptical_indecies: typing.Optional[np.ndarray] = None
        self.inflow_indecies: typing.Optional[np.ndarray] = None

        self.generated_boundary_information: bool = False

    def split_boundaries(self,
                         advection: typing.Callable[[np.ndarray], np.ndarray],
                         diffusion: typing.Callable[[np.ndarray], np.ndarray]) -> None:
        """
        This is the main method of this class. This generated the boudnary information from the geometry as well as the
        advection and diffusion coefficients.

        Args:
            advection (typing.Callable[[np.ndarray], np.ndarray]): The advection coefficient.
            diffusion (typing.Callable[[np.ndarray], np.ndarray]): The diffusion coefficient.

        Notes:
            - This function works under the assumption that for each edge of the mesh, the whole edge is categorised as
            the same type of boundary edge. This is defined as the behaviour at the midpoint of each edge. Changes in
            the advetion/diffusion fields away from this point are not taken into account.

        """
        bdmids = 0.5 * (self.geometry.nodes[self.geometry.boundary_edges[:, 0], :] +
                        self.geometry.nodes[self.geometry.boundary_edges[:, 1], :])

        if diffusion is not None:
            # indecies of the boundary edges that are hyperbolic in nature.
            hyp_index = np.einsum(
                'ni,nij,nj->n',
                self.geometry.boundary_normals,
                diffusion(bdmids),
                self.geometry.boundary_normals) <= self.tolerence

            self.elliptical_indecies = np.array(
                [index for index, v in enumerate(hyp_index) if not v]
            )

        if advection is not None:
            # indecies of the boundary edges that are inflow boundary edges.
            inflow_indecies = np.sum(advection(bdmids) * self.geometry.boundary_normals, 1) <= self.tolerence
            self.inflow_indecies = np.array([
                index for index, v in enumerate(inflow_indecies) if v
            ])

        self.generated_boundary_information = True

    def plot_boundaries(self) -> None:
        """
        This function generates a plot of the boundary edges of the domain, labelled with there respective
        classification.

        Raises:
            ValueError: If the boundary information has not been generated, this function can't be run. Please run the
            'split_boundaries' method first.

        """

        if not self.generated_boundary_information:
            raise ValueError('The boundary information has not been generated: please run the '
                             '"split_boundaries" method.')

        fig, ax = plt.subplots()

        if self.elliptical_indecies is not None:
            for i, ind in enumerate(self.elliptical_indecies):
                edge_vertices = self.geometry.nodes[self.geometry.boundary_edges[ind, :], :]

                if i == 0:
                    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], label='Elliptical Boundary', c='y')
                else:
                    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], c='y')

        if self.inflow_indecies is not None:
            for i, ind in enumerate(self.inflow_indecies):
                edge_vertices = self.geometry.nodes[self.geometry.boundary_edges[ind, :], :]

                if i == 0:
                    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], label='Inflow Boundary', c='g')
                else:
                    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], c='g')

        plt.legend()
        plt.show()

import typing

import numpy as np
import matplotlib.pyplot as plt

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry


class BoundaryInformation:
    """
    This object defines the boundary information for a PDE problem. This class takes in the mesh geometry as well as the
    information from the corresponding PDE and generates the boundary information from them. This is more for
    convienient storage of the information for use in the later numerical methods.
    """

    def __init__(self, **kwargs):

        self.tolerence = 1e-12
        if 'tolerence' in kwargs:
            self.tolerence = kwargs.pop('tolerence')

        self.elliptical_indecies: typing.Optional[np.ndarray] = None
        self.inflow_indecies: typing.Optional[np.ndarray] = None

        # self.elliptical_dirichlet_and_inflow_indecies: typing.Optional[np.ndarray] = None

        self.generated_boundary_information: bool = False

    def split_boundaries(self,
                         geometry: DGFEMGeometry,
                         advection: typing.Callable[[np.ndarray], np.ndarray],
                         diffusion: typing.Callable[[np.ndarray], np.ndarray]):

        bdmids = 0.5 * (geometry.nodes[geometry.boundary_edges[:, 0], :] +
                        geometry.nodes[geometry.boundary_edges[:, 1], :])

        self.inflow_indecies = np.array([], dtype=int)

        if diffusion is not None:
            # indecies of the boundary edges that are hyperbolic
            hyp_index = np.einsum(
                'ni,nij,nj->n',
                geometry.boundary_normals,
                diffusion(bdmids),
                geometry.boundary_normals) <= self.tolerence

            self.elliptical_indecies = np.array(
                [index for index, v in enumerate(hyp_index) if not v]
            )

        if advection is not None:
            inflow_indecies = np.sum(advection(bdmids) * geometry.boundary_normals, 1) <= self.tolerence
            self.inflow_indecies = np.array([
                i for i, v in enumerate(inflow_indecies) if v
            ])

        self.generated_boundary_information = True

    def plot_boundaries(self, geometry):

        if not self.generated_boundary_information:
            raise ValueError('The boundary information has not been generated: please run the '
                             '"split_boundaries" method.')

        fig, ax = plt.subplots()

        if self.elliptical_indecies is not None:
            for i, ind in enumerate(self.elliptical_indecies):
                edge_vertices = geometry.nodes[geometry.boundary_edges[ind, :], :]

                if i == 0:
                    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], label='Elliptical Boundary', c='y')
                else:
                    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], c='y')

        if self.inflow_indecies is not None:
            for i, ind in enumerate(self.inflow_indecies):
                edge_vertices = geometry.nodes[geometry.boundary_edges[ind, :], :]

                if i == 0:
                    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], label='Inflow Boundary', c='g')
                else:
                    ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], c='g')

        plt.legend()
        plt.show()

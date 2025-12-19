import typing

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry

from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import tensor_tensor_leg
from reyna.DGFEM.two_dimensional._auxilliaries.polygonal_basis_utils import Basis_index2D


def plot_DG(numerical_solution: np.ndarray, geometry: DGFEMGeometry, poly_degree: int) -> None:
    """
    This function plots the DGFEM solution using matplotlib.

    Args:
        numerical_solution (np.ndarray): The numerical solution of the DGFEM problem. This must be in the form that is
            outputted from the .dgfem method of the DGFEM class.
        geometry (DGFEMGeometry): The geometry of the DGFEM problem.
        poly_degree (int): The highest total degree polynomial space required.

    Returns:
        None

    Notes:
        - This function will be recieving large updates in a near update to allow significantly more customisation in
          the plots themselves as well as saving the plots.
    """

    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    U_min, U_max = np.inf, -np.inf

    fig, ax_poly = plt.subplots(subplot_kw={'projection': '3d'})

    if poly_degree == 0:
        for t in range(geometry.n_elements):
            elem = geometry.mesh.filtered_regions[t]
            nodes = geometry.nodes[elem, :]

            # create plot
            vtx = np.hstack((nodes, np.atleast_2d(np.array(nodes.shape[0] * [numerical_solution[t]])).T))
            tri = a3.art3d.Poly3DCollection([vtx])

            u_mean = np.mean(numerical_solution)
            U_max = np.max(numerical_solution)

            tri.set_facecolor([abs(u_mean / U_max), abs(u_mean / U_max), 1 - abs(u_mean / U_max)])

            x_min = np.minimum(x_min, np.min(nodes[:, 0]))
            x_max = np.maximum(x_max, np.max(nodes[:, 0]))
            y_min = np.minimum(y_min, np.min(nodes[:, 1]))
            y_max = np.maximum(y_max, np.max(nodes[:, 1]))

            tri.set_edgecolor('k')
            ax_poly.add_collection3d(tri)

            ax_poly.set_zlim(None, None)  # Auto-scale the z-axis based on data
    else:
        Lege_ind = Basis_index2D(poly_degree)
        dim_elem = Lege_ind.shape[0]

        cmap = cm.get_cmap('viridis')  # You can change 'viridis' to other colormaps like 'plasma', 'inferno', etc.
        for t in range(geometry.n_elements):
            elem, BDbox = geometry.mesh.filtered_regions[t], geometry.elem_bounding_boxes[t]
            node = geometry.nodes[elem, :]
            coef = numerical_solution[t * dim_elem: (t + 1) * dim_elem]

            u_DG_val = evaluate(node, coef, BDbox, orders=Lege_ind)

            U_max = np.maximum(U_max, np.max(u_DG_val))
            U_min = np.minimum(U_min, np.min(u_DG_val))

            x_min = np.minimum(x_min, np.min(node[:, 0]))
            x_max = np.maximum(x_max, np.max(node[:, 0]))
            y_min = np.minimum(y_min, np.min(node[:, 1]))
            y_max = np.maximum(y_max, np.max(node[:, 1]))

        for t in range(geometry.n_elements):

            elem, BDbox = geometry.mesh.filtered_regions[t], geometry.elem_bounding_boxes[t]
            node = geometry.nodes[elem, :]
            coef = numerical_solution[t * dim_elem: (t + 1) * dim_elem]

            u_DG_val = evaluate(node, coef, BDbox, orders=Lege_ind)

            # create plot
            vtx = np.hstack((node, u_DG_val[:, np.newaxis]))
            poly = a3.art3d.Poly3DCollection([np.array(vtx)])

            u_mean = np.mean(u_DG_val)
            scaling = max(np.abs(U_max), np.abs(U_min))

            poly.set_facecolor([
                abs(u_mean / scaling),
                abs(u_mean / scaling),
                1.0 - abs(u_mean / scaling)
            ])

            poly.set_edgecolor('k')
            ax_poly.add_collection3d(poly)

    ax_poly.set_xlim(x_min, x_max)
    ax_poly.set_ylim(y_min, y_max)
    ax_poly.set_zlim(U_min, U_max)

    # Construct the filename
    ax_poly.set_xlabel(r'$x$')
    ax_poly.set_ylabel(r'$y$')
    ax_poly.set_zlabel(r'$u$')

    plt.show()


def evaluate(x: np.ndarray,
             dg_coefficients: np.ndarray, bounding_box: np.ndarray,
             polynomial_degree: typing.Optional[int] = None,
             orders: typing.Optional[np.ndarray] = None) -> np.ndarray:
    """
    A simple element-wise evaluation function. Given a set of points, one may evalulate the DG solution at these points.
    Additionally, the bounding box of the element in question is required to specify the basis functions. This function
    also requires either the polynomial degree of the FEM space or the FEM orders. Either of these may be left blank
    but at the cost of additional computational overhead. Placing the orders in directly is the most efficient.

    Notes:
        The input points `x` must lie within the element in question (or on the boundary). If not this does not reflect
        the correct FEM solution.

    Args:
        x (np.ndarray): the points to evaluate.
        dg_coefficients (np.ndarray): The DG coefficients.
        bounding_box (np.ndarray): The bounding box of the element in question.
        polynomial_degree (typing.Optional[int]): The polynomial of the corresponding FEM space.
        orders (typing.Optional[np.ndarray]): The FEM indecies of the tensored polynomials.

    Returns:
        (np.ndarray): The finite element solution at the given basis elements.
    """

    if orders is None:
        if polynomial_degree is None:
            polynomial_degree = 0.5 * (-3.0 + np.sqrt(1.0 + 8.0 * dg_coefficients.shape[0]))

        orders = Basis_index2D(polynomial_degree)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    tensor_leg_array = tensor_tensor_leg(x, m, h, orders)

    return tensor_leg_array.T @ dg_coefficients

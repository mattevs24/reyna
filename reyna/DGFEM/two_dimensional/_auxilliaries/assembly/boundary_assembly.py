import typing
import time

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import tensor_tensor_leg, tensor_gradtensor_leg


def local_advection_inflow(nodes: np.ndarray,
                           bounding_box: np.ndarray,
                           edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                           orders: np.ndarray,
                           normal: np.ndarray,
                           advection: typing.Callable[[np.ndarray], np.ndarray],
                           dirichlet_bcs: typing.Callable[[np.ndarray], np.ndarray]) -> (np.ndarray, np.ndarray):
    """
    This function generates the information for the inflow boundary contribution to the advection term as well as
    the boundary value contribution to the forcing term. This is restricted to the inflow boundary.

    Args:
        nodes (np.ndarray): The vertices of the boundary edge in question.
        bounding_box (np.ndarray): The bounding box for the element in question
        edge_quadrature_rule (typing.Tuple[np.ndarray, np.ndarray]): The quadrature rule for the edge in question.
        orders (np.ndarray): The orders of the associated tensor-Legendre polynomials.
        normal (np.ndarray): The normal vector of the edge. The direction is pre-determined.
        advection (typing.Callable[[np.ndarray], np.ndarray]): The advection function.
        dirichlet_bcs (typing.Callable[[np.ndarray], np.ndarray]): The Dirichlet boundary conditions.
    Returns:
        (np.ndarray, np.ndarray): The local stiffness matrix and forcing vector associated with the local outflow
        boundary.
    """

    # Unpacking quadrature points and weights
    weights, ref_Qpoints = edge_quadrature_rule

    # Convert reference to physical quadrature points and calculate length value, De, and midpoint, mid_point
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    # Getting function values for the quadrature
    b_dot_n = np.sum(advection(P_Qpoints) * normal[None, :], axis=1)
    g_val = dirichlet_bcs(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    tensor_leg_array = tensor_tensor_leg(P_Qpoints, m, h, orders)

    _z = (b_dot_n * tensor_leg_array)  # Aux. term
    z = _z @ (weights * tensor_leg_array.T)  # Bilinear term
    z_f = (g_val * _z) @ weights  # Forcing term

    return De * z, De * z_f


def local_diffusion_dirichlet(nodes: np.ndarray,
                              bounding_box: np.ndarray,
                              edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                              orders: np.ndarray,
                              element_nodes: np.ndarray,
                              k_area: float, polydegree: float,
                              sigma_D: float,
                              normal: np.ndarray,
                              dirichlet_bcs: typing.Callable[[np.ndarray], np.ndarray],
                              diffusion: typing.Callable[[np.ndarray], np.ndarray]) -> (np.ndarray, np.ndarray):
    """
    This function calculates the contribution of enforcing the boundary conditions to the forcing linear function as
    well as the contribution to the local stiffness matrix through the interactions of the diffusion operator with the
    boundary.

    Args:
        nodes (np.ndarray): The vertices of the boundary edge in question.
        bounding_box (np.ndarray): The bounding box of the corresponding polygon.
        edge_quadrature_rule (typing.Tuple[np.ndarray, np.ndarray]): The quadrature rule for the edge in question.
        orders (np.ndarray): The orders of the associated tensor-Legendre polynomials.
        element_nodes (np.ndarray): The vertices of the element in question.
        k_area (float): The area of the element in question.
        polydegree (int): The max degree of Legendre polynomial used (this is used for the interior penalty terms).
        sigma_D (float): The global penalisation parameter.
        normal (np.ndarray): The OPUNV to the boundary edge.
        dirichlet_bcs (typing.Callable[[np.ndarray], np.ndarray]): The Dirichlet boundary conditions.
        diffusion (typing.Callable[[np.ndarray], np.ndarray]): The diffusion function.
    Returns:
        (np.ndarray, np.ndarray): The local stiffness matrix and forcing vector associated with the elliptic boundary.
    """

    # Unpacking quadrature points and weights
    weights, ref_Qpoints = edge_quadrature_rule

    # Convert reference to physical quadrature points and calculate length value, De, and midpoint, mid_point
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    weights = De * weights

    # Penalty term
    lambda_dot = normal @ diffusion(mid).squeeze() @ normal
    abs_k_b = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes - nodes[0, :]))))

    # Assuming p-coverability
    c_inv = min(k_area / abs_k_b, polydegree ** 2)
    sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * c_inv / k_area

    # Assuming not p-coverable
    # sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) / abs_k_b

    # Getting function values for the quadrature
    g_val = dirichlet_bcs(P_Qpoints)
    a_val = diffusion(P_Qpoints)

    coe = np.sum(a_val * normal[None, None, :], axis=2)

    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])
    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])

    tensor_leg_array = tensor_tensor_leg(P_Qpoints, m, h, orders)
    gradtensor_leg_array = tensor_gradtensor_leg(P_Qpoints, m, h, orders)

    wc = weights * coe

    Zx = ((gradtensor_leg_array[:, :, 0] * wc[:, 0]) @ tensor_leg_array.T +
          (tensor_leg_array * wc[:, 0]) @ gradtensor_leg_array[:, :, 0].T)
    Zy = ((gradtensor_leg_array[:, :, 1] * wc[:, 1]) @ tensor_leg_array.T +
          (tensor_leg_array * wc[:, 1]) @ gradtensor_leg_array[:, :, 1].T)
    Zs = -sigma * (tensor_leg_array * weights.T) @ tensor_leg_array.T

    z = Zx + Zy + Zs

    z_f = np.dot(g_val * (np.sum(coe * gradtensor_leg_array, axis=-1) - sigma * tensor_leg_array), weights)

    return z, z_f

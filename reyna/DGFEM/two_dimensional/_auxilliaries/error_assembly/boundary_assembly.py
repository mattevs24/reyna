import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import tensor_tensor_leg


def error_d_face(nodes: np.ndarray,
                 bounding_box: np.ndarray,
                 edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                 orders: np.ndarray,
                 element_nodes: np.ndarray,
                 k_area: float,
                 polydegree: float,
                 sigma_D: float,
                 dg_coefs: np.ndarray,
                 normal: np.ndarray,
                 u_exact: typing.Callable[[np.ndarray], np.ndarray],
                 diffusion: typing.Callable[[np.ndarray], np.ndarray]) -> float:

    """
    This function calculates the DG sub-norm, sigma [[u-u_h]]**2, over a boundary facet.

    Args:
        nodes (np.ndarray): The vertices of the boundary edge in question.
        bounding_box (np.ndarray): The bounding box of the corresponding polygon.
        edge_quadrature_rule (typing.Tuple[np.ndarray, np.ndarray]): The quadrature rule for the edge in question.
        orders (np.ndarray): The orders of the associated tensor-Legendre polynomials.
        element_nodes (np.ndarray): The vertices of the element in question.
        k_area (float): The area of the element in question.
        polydegree (int): The max degree of Legendre polynomial used (this is used for the interior penalty terms).
        sigma_D (float): The global penalisation parameter.
        dg_coefs (np.ndarray): The coefficient vector of the solution vector for the element in question.
        normal (np.ndarray): The OPUNV to the boundary edge.
        u_exact (typing.Callable[[np.ndarray], np.ndarray]): The true solution function.
        diffusion (typing.Callable[[np.ndarray], np.ndarray]): The diffusion function.
    Returns:
        (float): The dg sub-norm associated with the diffusion term on a boundary facet.
    """

    # Generate the reference domain quadrature points
    weights, ref_Qpoints = edge_quadrature_rule

    # Change the quadrature from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    # Penalty term
    lambda_dot = normal @ diffusion(mid).squeeze() @ normal

    abs_k_b = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes - nodes[0, :]))))

    # Assuming p-coverability
    c_inv = min(k_area / abs_k_b, polydegree ** 2)
    sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * c_inv / k_area

    # Assuming not p-coverable
    # sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) / abs_k_b

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    # Getting function values for the quadrature
    u_val = u_exact(P_Qpoints)

    # Getting dG solutions values for the quadrature
    tensor_leg_array = tensor_tensor_leg(P_Qpoints, m, h, orders)
    u_DG_val = tensor_leg_array.T @ dg_coefs

    t = sigma * (u_val - u_DG_val) ** 2
    dg_subnorm = De * np.dot(t, weights)[0]

    return dg_subnorm


def error_a_face(nodes: np.ndarray,
                 bounding_box: np.ndarray,
                 edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                 orders: np.ndarray,
                 dg_coefs: np.ndarray,
                 normal: np.ndarray,
                 u_exact: typing.Callable[[np.ndarray], np.ndarray],
                 advection: typing.Callable[[np.ndarray], np.ndarray]) -> float:

    """
    This function calculates the DG sub-norm, (u-u_h)^+, over a boundary facet.

    Args:
        nodes (np.ndarray): The vertices of the boundary edge in question.
        bounding_box (np.ndarray): The bounding box of the corresponding polygon.
        edge_quadrature_rule (typing.Tuple[np.ndarray, np.ndarray]): The quadrature rule for the edge in question.
        orders (np.ndarray): The orders of the associated tensor-Legendre polynomials.
        dg_coefs (np.ndarray): The coefficient vector of the solution vector for the element in question.
        normal (np.ndarray): The OPUNV to the boundary edge.
        u_exact (typing.Callable[[np.ndarray], np.ndarray]): The true solution function.
        advection (typing.Callable[[np.ndarray], np.ndarray]): The diffusion function.
    Returns:
        (float): The dg sub-norm associated with the advection term on a boundary facet.
    """

    # Generate the reference domain quadrature points
    weights, ref_Qpoints = edge_quadrature_rule

    # Change the quadrature from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    # Getting function values for the quadrature
    b_val = advection(P_Qpoints)
    n_vec = np.kron(normal, np.ones((ref_Qpoints.shape[0], 1)))
    u_val = u_exact(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    # Getting dG solutions values for the quadrature
    tensor_leg_array = tensor_tensor_leg(P_Qpoints, m, h, orders)
    u_DG_val = tensor_leg_array.T @ dg_coefs

    t = 0.5 * np.abs(np.sum(b_val * n_vec, axis=1)) * (u_val - u_DG_val) ** 2
    dg_subnorm = De * np.dot(t, weights)[0]

    return dg_subnorm

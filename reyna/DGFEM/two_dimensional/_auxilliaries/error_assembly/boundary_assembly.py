import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.assembly_aux import tensor_leg


def error_bd_face(nodes: np.ndarray,
                  bounding_box: np.ndarray,
                  df_coefs: np.ndarray,
                  normal: np.ndarray,
                  edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                  Lege_ind: np.ndarray,
                  u_exact: typing.Callable[[np.ndarray], np.ndarray],
                  diffusion: typing.Callable[[np.ndarray], np.ndarray],
                  element_nodes: np.ndarray,
                  k_area: float, polydegree: float,
                  sigma_D: float) -> float:

    """
    This function calculates the DG sub-norm, sigma [[u-u_h]]**2, over a boundary facet.

    Args:
        nodes: The nodes of the simplex in question.
        bounding_box: The bounding box of the element which contains the simplex.
        df_coefs: The DG coefficients corresponding to the element in question.
        normal: The OPUNV to the boundary facet.
        edge_quadrature_rule: The required quadrature rule.
        Lege_ind: The indecies of the tensored Legendre polynomials required (must match with dg_coefs).
        u_exact: The true solution to the PDE.
        diffusion: The diffusion component to the PDE.
        sigma_D: The diffusion penalty parameter.

    Returns:
        (float): The DG sub-norm over the boundary facet in question.
    """

    # Calculate the local stiffness matrix

    dim_elem = Lege_ind.shape[0]  # number of basis for each element

    # Generate the reference domain quadrature points
    weights, ref_Qpoints = edge_quadrature_rule

    # Change the quadrature from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    # penalty term
    lambda_dot = normal @ diffusion(mid[None, :]).squeeze() @ normal

    abs_k_b = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes - nodes[0, :]))))
    c_inv = min(k_area / abs_k_b, polydegree ** 2)
    sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * c_inv / k_area

    u_val = u_exact(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    tensor_leg_array = np.array([
        tensor_leg(P_Qpoints, m, h, Lege_ind[i, :]) for i in range(dim_elem)
    ])

    u_DG_val = tensor_leg_array.T @ df_coefs

    t = sigma * (u_val - u_DG_val) ** 2
    dg_subnorm = De * np.dot(t, weights)[0]

    return dg_subnorm


def error_cr_bd_face(nodes: np.ndarray,
                     bounding_box: np.ndarray,
                     df_coefs: np.ndarray,
                     normal: np.ndarray,
                     edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                     Lege_ind: np.ndarray,
                     u_exact: typing.Callable[[np.ndarray], np.ndarray],
                     advection: typing.Callable[[np.ndarray], np.ndarray]):

    """
    This function calculates the DG sub-norm, (u-u_h)^+, over a boundary facet.

    Args:
        nodes: The nodes of the simplex in question.
        bounding_box: The bounding box of the element.
        df_coefs: The DG coefficients corresponding to the element in question.
        normal: The OPUNV to the boundary facet.
        edge_quadrature_rule: The required quadrature rule.
        Lege_ind: The indecies of the tensored Legendre polynomials required (must match with dg_coefs).
        u_exact: The true solution to the PDE.
        advection: The advection component to the PDE.

    Returns:
        (float): The DG sub-norm over the boundary facet in question.
    """

    dim_elem = Lege_ind.shape[0]  # number of basis for each element

    # Generate the reference domain quadrature points
    weights, ref_Qpoints = edge_quadrature_rule

    # Change the quadrature from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    # data for quadrature, function value b and normal vector n_vec
    b_val = advection(P_Qpoints)
    n_vec = np.kron(normal, np.ones((ref_Qpoints.shape[0], 1)))
    u_val = u_exact(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    tensor_leg_array = np.array([
        tensor_leg(P_Qpoints, m, h, Lege_ind[i, :]) for i in range(dim_elem)
    ])

    u_DG_val = tensor_leg_array.T @ df_coefs

    # Part 3 DG norm error  int_\e  1/2*|b \cdot n |*(u-u_DG)^2 dx
    t = 0.5 * np.abs(np.sum(b_val * n_vec, axis=1)) * (u_val - u_DG_val) ** 2
    dg_subnorm = De * np.dot(t, weights)[0]

    return dg_subnorm

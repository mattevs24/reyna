import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import tensor_tensor_leg, tensor_gradtensor_leg


def local_advection_inflow(nodes: np.ndarray,
                           bounding_box: np.ndarray,
                           normal: np.ndarray,
                           edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                           Lege_ind: np.ndarray,
                           advection: typing.Callable[[np.ndarray], np.ndarray],
                           dirichlet_bcs: typing.Callable[[np.ndarray], np.ndarray]) -> (np.ndarray, np.ndarray):
    """
    This function generates the information for the inflow boundary contribution to the advection term as well as
    the boundary value contribution to the forcing term. This is restricted to the inflow boundary.

    Args:
        nodes: The endpoints/vertieces of the boundary edge in question.
        bounding_box: The bounding box of the polygon in question.
        normal: the OPUNV to the boundary edge.
        edge_quadrature_rule: The quadrature rule for the edge in question.
        Lege_ind: The indecies of the tensor Legendre polynomials.
        advection: The advection function.
        dirichlet_bcs: The Dirichlet boundary conditions required for the problem.
    Returns:
        (np.ndarray, np.ndarray): The local stiffness matrix associated with the local outflow boundary.
    """
    weights, ref_Qpoints = edge_quadrature_rule

    # change the quarature nodes from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    b_dot_n = np.sum(advection(P_Qpoints) * normal[None, :], axis=1)
    g_val = dirichlet_bcs(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    tensor_leg_array = tensor_tensor_leg(P_Qpoints, m, h, Lege_ind)

    _z = (b_dot_n * tensor_leg_array)
    z = _z @ (weights * tensor_leg_array.T)  # Bilinear term
    z_f = (g_val * _z) @ weights  # Forcing term

    return De * z, De * z_f


def local_diffusion_dirichlet(nodes: np.ndarray,
                              bounding_box: np.ndarray,
                              normal: np.ndarray,
                              edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                              Lege_ind: np.ndarray,
                              element_nodes: np.ndarray,
                              k_area: float, polydegree: float,
                              sigma_D: float,
                              dirichlet_bcs: typing.Callable[[np.ndarray], np.ndarray],
                              diffusion: typing.Callable[[np.ndarray], np.ndarray]) -> (np.ndarray, np.ndarray):
    """
    This function calculates the contribution of enforcing the boundary conditions to the forcing linear function as
    well as the contribution to the local stiffness matrix through the interactions of the diffusion operator with the
    boundary.

    Args:
        nodes: The endpoints of the boundary edge.
        bounding_box: The bounding box of the corresponding polygon.
        normal: The OPUNV to the boundary edge.
        edge_quadrature_rule: The quadrature rule for the edge in question.
        Lege_ind: The polynomial indecies in question.
        element_nodes: The nodes of the element in question to generate the penalty term
        k_area: the area of the element in question
        polydegree: the polynomial degree of the system
        sigma_D: The global penalty parameter.
        dirichlet_bcs: The dirichlet boundary conditions to be enforced.
        diffusion: The diffusion coefficient (tensor) function.
    Returns:
        (np.ndarray, np.ndarray): The 1D array containing the contributions of each of the basis functions as well as
        the local stiffness matrix contributions.
    """

    # quadrature data
    weights, ref_Qpoints = edge_quadrature_rule

    # change the quadrature nodes from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    # penalty term
    lambda_dot = normal @ diffusion(mid[None, :]).squeeze() @ normal
    abs_k_b = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes - nodes[0, :]))))

    # Assuming p-coverability
    # c_inv = min(k_area / abs_k_b, polydegree ** 2)
    # sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * c_inv / k_area

    # Assuming lack of p-coverability
    sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) / abs_k_b

    g_val = dirichlet_bcs(P_Qpoints)
    a_val = diffusion(P_Qpoints)

    coe = np.sum(a_val * normal[None, None, :], axis=2)

    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])
    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])

    tensor_leg_array = tensor_tensor_leg(P_Qpoints, m, h, Lege_ind)
    gradtensor_leg_array = tensor_gradtensor_leg(P_Qpoints, m, h, Lege_ind)

    dim_elem = Lege_ind.shape[0]

    z = np.zeros((dim_elem, dim_elem))

    for i in range(dim_elem):
        for j in range(i, dim_elem):

            t = (coe[:, 0] * (gradtensor_leg_array[i, :, 0] * tensor_leg_array[j, :] +
                             gradtensor_leg_array[j, :, 0] * tensor_leg_array[i, :]) +
                coe[:, 1] * (gradtensor_leg_array[i, :, 1] * tensor_leg_array[j, :] +
                             gradtensor_leg_array[j, :, 1] * tensor_leg_array[i, :]) -
                 sigma * tensor_leg_array[i, :] * tensor_leg_array[j, :])
            z[j, i] = np.dot(t, weights)

    z += z.T - np.diag(np.diag(z))

    z_f = np.dot(g_val * (np.sum(coe * gradtensor_leg_array, axis=-1) - sigma * tensor_leg_array), weights)

    return De * z, De * z_f

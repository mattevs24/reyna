import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.assembly_aux import tensor_leg, gradtensor_leg


def C_vecDiriface(nodes: np.ndarray,
                  bounding_box: np.ndarray,
                  normal: np.ndarray,
                  edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                  Lege_ind: np.ndarray,
                  element_nodes: np.ndarray,
                  k_area: float, polydegree: float,
                  sigma_D: float,
                  dirichlet_bcs: typing.Callable[[np.ndarray], np.ndarray],
                  diffusion: typing.Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    This function calculates the contribution of enforcing the boundary conditions to the forcing linear function.

    Args:
        nodes: The endpoints of the boundary edge.
        bounding_box: The bounding box of the corresponding polygon.
        normal: The OPUNV to the boundary edge.
        edge_quadrature_rule: The quadrature rule for the edge in question.
        Lege_ind: The polynomial indecies in question.
        sigma_D: The global penalty parameter.
        dirichlet_bcs: The dirichlet boundary conditions to be enforced.
        diffusion: The diffusion coefficient (tensor) function.
    Returns:
        (np.ndarray): The 1D array containing the contributions of each of the basis functions.
    """
    # Calculate the local stiffness matrix
    dim_elem = Lege_ind.shape[0]

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
    c_inv = min(k_area / abs_k_b, polydegree ** 2)
    sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * c_inv / k_area

    # n_vec = np.kron(normal, np.ones((ref_Qpoints.shape[0], 1)))
    g_val = dirichlet_bcs(P_Qpoints)
    a_val = diffusion(P_Qpoints)

    coe = np.einsum('ijk,ik->ij', a_val, normal[None, :] * np.ones((ref_Qpoints.shape[0], 1)))

    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])
    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])

    z = np.zeros(dim_elem)

    for j in range(dim_elem):

        V = tensor_leg(P_Qpoints, m, h, Lege_ind[j, :])
        gradv = gradtensor_leg(P_Qpoints, m, h, Lege_ind[j, :])

        to_be_dot = coe[:, 0] * gradv[:, 0] + coe[:, 1] * gradv[:, 1] - sigma * V
        t = g_val * to_be_dot

        z[j] = np.dot(t, weights)

    return De * z


def vect_inflowDiriface(nodes: np.ndarray,
                        normal: np.ndarray,
                        bounding_box: np.ndarray,
                        edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                        Lege_ind: np.ndarray,
                        advection: typing.Callable[[np.ndarray], np.ndarray],
                        dirichlet_bcs: typing.Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    This function calculates the boundary value contribution to the forcing term. This is restricted to the inflow
    boundary.

    Args:
        nodes: The endpoints/vertices of the edge in question.
        normal: The OPUNV to the boundary edge in question.
        bounding_box: The boundung box to the element to which the edge belongs.
        edge_quadrature_rule: The quadrature rule of the edge in question
        Lege_ind: The indecies of the tensor Legendre polynomials in question.
        advection: The advection field for the problem.
        dirichlet_bcs: The Dirichlet boundary conditions required for the problem.
    """
    # Calcualte the local stiffness matrix

    dim_elem = Lege_ind.shape[0]  # number of basis for each element

    # quadrature data
    weights, ref_Qpoints = edge_quadrature_rule

    # change the quadrature nodes from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    # Data for quadrature
    z = np.zeros(dim_elem)

    b_dot_n = np.sum(advection(P_Qpoints) * normal[None, :], axis=1)
    g_val = dirichlet_bcs(P_Qpoints)

    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])
    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])

    # i is the row which is the basis of u and j is of v
    for j in range(dim_elem):
        V = tensor_leg(P_Qpoints, m, h, Lege_ind[j, :])

        t = b_dot_n * g_val * V
        z[j] = np.dot(t, weights)

    return De * z

import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.assembly_aux import tensor_leg, gradtensor_leg


def localstiff_diffusion_bcs(nodes: np.ndarray,
                             normal: np.ndarray,
                             bounding_box: np.ndarray,
                             edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                             Lege_ind: np.ndarray,
                             element_nodes: np.ndarray,
                             k_area: float, polydegree: float,
                             sigma_D: float,
                             diffusion: typing.Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    This function calculates the contribution to the local stiffness matrix through the interactions of the diffusion
    operator with the boundary.

    Args:
        nodes: The two boundary nodes between which the edge lies
        normal: The OPUNV to the boundary edge in question
        bounding_box: The bounding box of the element in question
        edge_quadrature_rule: The quadrature rule for the edge
        Lege_ind: The indecies of the monomials in question to be integrated over the edge
        sigma_D: The global penalty parameter
        diffusion: The diffusion operator of the system
    Returns:
        (np.ndarray): The local stiffness matrix. Note that this lives in the monomial basis and needs to be projected
        to the Legendre polynomial basis space.
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

    # n_vec = np.kron(normal, np.ones((ref_Qpoints.shape[0], 1)))
    a_val = diffusion(P_Qpoints)
    coe = np.einsum('ijk,ik->ij', a_val, normal[None, :] * np.ones((ref_Qpoints.shape[0], 1)))

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    z = np.zeros((dim_elem, dim_elem))

    tensor_leg_cache = {}
    gradtensor_leg_cache = {}

    for i in range(dim_elem):

        if i not in tensor_leg_cache:
            tensor_leg_cache[i] = tensor_leg(P_Qpoints, m, h, Lege_ind[i, :])
            gradtensor_leg_cache[i] = gradtensor_leg(P_Qpoints, m, h, Lege_ind[i, :])

        U = tensor_leg_cache[i]
        gradu = gradtensor_leg_cache[i]

        for j in range(i, dim_elem):

            if j not in tensor_leg_cache:
                tensor_leg_cache[j] = tensor_leg(P_Qpoints, m, h, Lege_ind[j, :])
                gradtensor_leg_cache[j] = gradtensor_leg(P_Qpoints, m, h, Lege_ind[j, :])

            V = tensor_leg_cache[j]
            gradv = gradtensor_leg_cache[j]

            t = coe[:, 0] * (gradu[:, 0] * V + gradv[:, 0] * U) + \
                coe[:, 1] * (gradu[:, 1] * V + gradv[:, 1] * U) - sigma * U * V
            z[j, i] = np.dot(t, weights)

    z += z.T - np.diag(np.diag(z))

    return De * z

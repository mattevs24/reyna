import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.assembly_aux import tensor_leg


def localinflowface(nodes: np.ndarray,
                    bounding_box: np.ndarray,
                    normal: np.ndarray,
                    edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                    Lege_ind: np.ndarray,
                    advection: typing.Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    This function generates the information for the outflow boundary contribution to the advection term.

    Args:
        nodes: The endpoints/vertieces of the boundary edge in question.
        bounding_box: The bounding box of the polygon in question.
        normal: the OPUNV to the boundary edge.
        edge_quadrature_rule: The quadrature rule for the edge in question.
        Lege_ind: The indecies of the tensor Legendre polynomials.
        advection: The advection function.
    Returns:
        (np.ndarray): The local stiffness matrix associated with the local outflow boundary.
    """

    dim_elem = Lege_ind.shape[0]

    weights, ref_Qpoints = edge_quadrature_rule

    # change the quarature nodes from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    b_dot_n = np.sum(advection(P_Qpoints) * normal[None, :], axis=1)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    z = np.zeros((dim_elem, dim_elem))

    tensor_leg_cache = {}

    for i in range(dim_elem):

        if i not in tensor_leg_cache:
            tensor_leg_cache[i] = tensor_leg(P_Qpoints, m, h, Lege_ind[i, :])

        U = tensor_leg_cache[i]

        for j in range(i, dim_elem):
            if j not in tensor_leg_cache:
                tensor_leg_cache[j] = tensor_leg(P_Qpoints, m, h, Lege_ind[j, :])

            V = tensor_leg_cache[j]

            t = b_dot_n * (U * V)
            z[j, i] = np.dot(t, weights)

    # Symmetric
    z += z.T - np.diag(np.diag(z))

    return De * z

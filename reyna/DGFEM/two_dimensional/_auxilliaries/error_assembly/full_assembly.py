import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.assembly_aux import tensor_leg, gradtensor_leg,\
    reference_to_physical_t3


def error_element(nodes: np.ndarray,
                  bounding_box: np.ndarray,
                  element_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                  Lege_ind: np.ndarray,
                  dg_coefs: np.ndarray,
                  u_exact: typing.Callable[[np.ndarray], np.ndarray],
                  diffusion: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
                  grad_u_exact: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
                  auxilliary_function: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None):

    if (diffusion is None) != (grad_u_exact is None):
        raise ValueError('Need to input both or neither "diffusion" and "grad_u_exact".')

    if auxilliary_function is None:
        auxilliary_function = lambda x: 0.0

    weights, ref_points = element_quadrature_rule
    B = 0.5 * np.vstack((nodes[1, :] - nodes[0, :], nodes[2, :] - nodes[0, :]))
    De_tri = np.abs(np.linalg.det(B))

    weights = De_tri * weights

    # The physical points
    P_Qpoints = reference_to_physical_t3(nodes, ref_points)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    u_val = u_exact(P_Qpoints)
    c0_val = auxilliary_function(P_Qpoints)

    dim_elem = Lege_ind.shape[0]

    tensor_leg_array = np.array([
        tensor_leg(P_Qpoints, m, h, Lege_ind[i, :]) for i in range(dim_elem)
    ])

    u_DG_val = tensor_leg_array.T @ dg_coefs

    t2, t4 = 0.0, 0.0

    if diffusion is not None:

        a_val = diffusion(P_Qpoints)
        a_val = a_val.reshape(a_val.shape[0], a_val.shape[1] * a_val.shape[2])
        grad_u_val = grad_u_exact(P_Qpoints)

        gradtensor_leg_array = np.array([
            gradtensor_leg(P_Qpoints, m, h, Lege_ind[i, :]) for i in range(dim_elem)
        ])

        grad_u_DG = np.vstack((gradtensor_leg_array[:, :, 0].T @ dg_coefs,
                               gradtensor_leg_array[:, :, 1].T @ dg_coefs)).T

        grad = grad_u_val - grad_u_DG
        t4 = np.sum(grad ** 2, axis=1)

        grad_11 = grad[:, 0] * grad[:, 0]
        grad_12 = grad[:, 0] * grad[:, 1]
        grad_21 = grad[:, 1] * grad[:, 0]
        grad_22 = grad[:, 1] * grad[:, 1]

        grad = np.hstack((grad_11[:, np.newaxis], grad_12[:, np.newaxis],
                          grad_21[:, np.newaxis], grad_22[:, np.newaxis]))

        t2 = np.sum(grad * a_val, axis=1)

    t1 = (u_val - u_DG_val) ** 2
    t3 = c0_val * t1

    l2_subnorm: float = 0.5 * np.dot(t1, weights)[0]
    dg_subnorm: float = 0.5 * np.dot(t2 + t3, weights)[0]
    h1_subnorm: float = 0.5 * np.dot(t4, weights)[0]

    if diffusion is None:
        return l2_subnorm, dg_subnorm, None

    return l2_subnorm, dg_subnorm, h1_subnorm


def error_interface(nodes: np.ndarray,
                    BDbox1: np.ndarray, BDbox2: np.ndarray,
                    edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                    Lege_ind: np.ndarray,
                    element_nodes_1: np.ndarray, element_nodes_2: np.ndarray,
                    k_1_area: float, k_2_area: float, polydegree: float,
                    sigma_D: float,
                    normal: np.ndarray,
                    dg_coefs1: np.ndarray, dg_coefs2: np.ndarray,
                    diffusion: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
                    advection: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]):

    # Information for two bounding box 1,2 n is the normal vector from k1 to k2
    h1 = 0.5 * np.array([BDbox1[1] - BDbox1[0], BDbox1[3] - BDbox1[2]])
    m1 = 0.5 * np.array([BDbox1[1] + BDbox1[0], BDbox1[3] + BDbox1[2]])
    h2 = 0.5 * np.array([BDbox2[1] - BDbox2[0], BDbox2[3] - BDbox2[2]])
    m2 = 0.5 * np.array([BDbox2[1] + BDbox2[0], BDbox2[3] + BDbox2[2]])

    dim_elem = Lege_ind.shape[0]  # number of basis for each element

    # Generating quadrature points and weights
    weights, ref_Qpoints = edge_quadrature_rule

    # Change the quadrature nodes from reference domain to physical domain
    mid = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    weights = De * weights

    tensor_leg_array = np.array([
        np.array([
            tensor_leg(P_Qpoints, m1, h1, Lege_ind[i, :]),
            tensor_leg(P_Qpoints, m2, h2, Lege_ind[i, :])
        ])
        for i in range(dim_elem)
    ])

    u_DG_val1 = np.matmul(tensor_leg_array[:, 0, :].T, dg_coefs1)  # DG solution on kappa1
    u_DG_val2 = np.matmul(tensor_leg_array[:, 1, :].T, dg_coefs2)  # DG solution on kappa2

    dg_subnorm: float = 0.0

    if diffusion is not None:
        lambda_dot = normal @ diffusion(mid[None, :]).squeeze() @ normal

        abs_k_b_1 = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes_1 - nodes[0, :]))))
        abs_k_b_2 = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes_2 - nodes[0, :]))))

        c_inv_1 = min(k_1_area / abs_k_b_1, polydegree ** 2)
        c_inv_2 = min(k_2_area / abs_k_b_2, polydegree ** 2)

        sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * max(c_inv_1 / k_1_area, c_inv_2 / k_2_area)

        t = (u_DG_val1 - u_DG_val2) ** 2
        dg_subnorm += sigma * np.dot(t, weights)[0]

    if advection is not None:
        b_val = advection(P_Qpoints)
        t = np.abs(np.sum(b_val * normal[None, :], axis=1)) * (u_DG_val1 - u_DG_val2) ** 2
        dg_subnorm += 0.5 * np.dot(t, weights)[0]

    return dg_subnorm

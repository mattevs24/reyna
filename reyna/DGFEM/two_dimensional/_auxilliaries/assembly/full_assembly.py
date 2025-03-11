import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly.assembly_aux import tensor_leg, \
    gradtensor_leg, reference_to_physical_t3


def localstiff(nodes: np.ndarray,
               bounding_box: np.ndarray,
               element_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
               Lege_ind: np.ndarray,
               diffusion: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
               advection: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
               reaction: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
               forcing: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]):
    """
    This function calculates the local stiffness matrix associated to the diffusion component of the PDE.

    Args:
        nodes: The vertices of the triangle in question
        bounding_box: The bounding box for the element in question
        element_quadrature_rule: The quadrature rule for the triangle in question
        Lege_ind: The indecies of the associated monomials
        diffusion: The diffusion function
        advection: The advection function
        reaction: The reaction function
        forcing: The forcing function
    Returns:
        (np.ndarray): Local stiffness matrix. Note that this stiffness matrix is in terms of the
        standard monomial basis and will be mapped to the Legendre polynomial space via a projection operator
    """

    weights, ref_points = element_quadrature_rule
    # Jacobian calculation
    B = 0.5 * np.vstack((nodes[1, :] - nodes[0, :], nodes[2, :] - nodes[0, :]))
    De_tri = np.abs(np.linalg.det(B))

    weights = De_tri * weights

    # The physical points
    P_Qpoints = reference_to_physical_t3(nodes, ref_points)

    dim_elem = Lege_ind.shape[0]  # Number of basis for each element
    z = np.zeros((dim_elem, dim_elem))

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    # Generate the tensor Legendre polynomial values
    tensor_leg_array = np.array([
        tensor_leg(P_Qpoints, m, h, Lege_ind[i, :]) for i in range(dim_elem)
    ])

    gradtensor_leg_array = np.array([
        gradtensor_leg(P_Qpoints, m, h, Lege_ind[i, :]) for i in range(dim_elem)
    ])
    if diffusion is not None:
        a_val = diffusion(P_Qpoints)
        z += np.einsum(
            'ikl,jkl->ij',
            np.einsum('kni,nij->knj', gradtensor_leg_array, a_val),
            weights[None, ...] * gradtensor_leg_array
        )

    if advection is not None:
        b_val = advection(P_Qpoints)
        z += np.sum(b_val[None, ...] * gradtensor_leg_array, axis=-1) @ (tensor_leg_array * weights.T).T

    if reaction is not None:
        c_val = reaction(P_Qpoints)
        z += (c_val * tensor_leg_array) @ (weights * tensor_leg_array.T)

    if forcing is not None:
        f_val = forcing(P_Qpoints)
        z_f = 0.5 * np.sum(f_val * tensor_leg_array * weights.T, axis=-1)
        return 0.5 * z, z_f

    return 0.5 * z, None


def int_localstiff(nodes: np.ndarray,
                   BDbox1: np.ndarray, BDbox2: np.ndarray,
                   edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                   Lege_ind: np.ndarray,
                   element_nodes_1: np.ndarray, element_nodes_2: np.ndarray,
                   k_1_area: float, k_2_area: float, polydegree: float,
                   sigma_D: float,
                   normal: np.ndarray,
                   diffusion: typing.Callable[[np.ndarray], np.ndarray],
                   advection: typing.Callable[[np.ndarray], np.ndarray]):

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

    tensor_leg_array = np.array([
        np.array([
            tensor_leg(P_Qpoints, m1, h1, Lege_ind[i, :]),
            tensor_leg(P_Qpoints, m2, h2, Lege_ind[i, :])
        ])
        for i in range(dim_elem)
    ])

    z = np.zeros((2 * dim_elem, 2 * dim_elem))

    if diffusion is not None:
        # penalty term
        lambda_dot = normal @ diffusion(mid[None, :]).squeeze() @ normal

        abs_k_b_1 = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes_1 - nodes[0, :]))))
        abs_k_b_2 = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes_2 - nodes[0, :]))))

        c_inv_1 = min(k_1_area / abs_k_b_1, polydegree ** 2)
        c_inv_2 = min(k_2_area / abs_k_b_2, polydegree ** 2)

        sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * max(c_inv_1 / k_1_area, c_inv_2 / k_2_area)

        auxiliary_sigma_1 = np.zeros((dim_elem, dim_elem))
        auxiliary_sigma_2 = np.zeros((dim_elem, dim_elem))
        auxiliary_sigma_3 = np.zeros((dim_elem, dim_elem))
        auxiliary_sigma_4 = np.zeros((dim_elem, dim_elem))

        gradtensor_leg_array = np.array([
            np.array([
                gradtensor_leg(P_Qpoints, m1, h1, Lege_ind[i, :]),
                gradtensor_leg(P_Qpoints, m2, h2, Lege_ind[i, :])
            ])
            for i in range(dim_elem)
        ])

        a_val = diffusion(P_Qpoints)

        a_gradx_array = np.einsum('ijk,nij -> nij', a_val, gradtensor_leg_array[:, 0])
        a_grady_array = np.einsum('ijk,nij -> nij', a_val, gradtensor_leg_array[:, 1])

        for i in range(dim_elem):

            U1, U2 = tensor_leg_array[i, 0][:, None], tensor_leg_array[i, 1][:, None]

            for j in range(i, dim_elem):

                V1, V2 = tensor_leg_array[j, 0][:, None], tensor_leg_array[j, 1][:, None]

                # put it into the matrix
                s1 = -sigma * (U1 * V1)
                t1 = 0.5 * (a_gradx_array[i, ...] * V1 + a_gradx_array[j, ...] * U1) @ normal[:, None]
                auxiliary_sigma_1[j, i] = np.dot((s1 + t1).T, weights)

                s2 = sigma * (U2 * V1)
                t2 = 0.5 * (a_grady_array[i, ...] * V1 - a_gradx_array[j, ...] * U2) @ normal[:, None]
                auxiliary_sigma_2[j, i] = np.dot((s2 + t2).T, weights)

                s3 = sigma * (U1 * V2)
                t3 = 0.5 * (-a_gradx_array[i, ...] * V2 + a_grady_array[j, ...] * U1) @ normal[:, None]
                auxiliary_sigma_3[j, i] = np.dot((s3 + t3).T, weights)

                s4 = -sigma * (U2 * V2)
                t4 = 0.5 * (-a_grady_array[i, ...] * V2 - a_grady_array[j, ...] * U2) @ normal[:, None]
                auxiliary_sigma_4[j, i] = np.dot((s4 + t4).T, weights)

        # term may be symmtric or skew-symmetric
        local_1 = auxiliary_sigma_1 + np.tril(auxiliary_sigma_1, -1).T  # U1, V1
        local_2 = auxiliary_sigma_2 + np.tril(auxiliary_sigma_3, -1).T  # U2, V1
        local_3 = auxiliary_sigma_3 + np.tril(auxiliary_sigma_2, -1).T  # U1, V2
        local_4 = auxiliary_sigma_4 + np.tril(auxiliary_sigma_4, -1).T  # U2, V2

        z -= np.vstack((np.hstack((local_1.T, local_3.T)), np.hstack((local_2.T, local_4.T))))

    if advection is not None:
        # Correct the normal vector's direction if required.
        correction = np.sum(advection(mid).flatten() * normal) >= 1e-12
        b_dot_n = np.sum(advection(P_Qpoints) * normal[None, :], axis=1)

        if correction:
            # This line allows V to be tensor_leg_array[j, 0] no matter what and U1 and U2 to be in that order always
            correction_indecies = [1, 0]
            b_dot_n *= -1
        else:
            correction_indecies = [0, 1]

        z_1 = -((b_dot_n * tensor_leg_array[:, correction_indecies[0]]) @
                (weights * tensor_leg_array[:, correction_indecies[0]].T)).T
        z_2 = ((b_dot_n * tensor_leg_array[:, correction_indecies[0]]) @
               (weights * tensor_leg_array[:, correction_indecies[1]].T)).T

        if correction:
            z[:, dim_elem:] += np.vstack((z_2, z_1))
        else:
            z[:, :dim_elem] += np.vstack((z_1, z_2))

    return De * z

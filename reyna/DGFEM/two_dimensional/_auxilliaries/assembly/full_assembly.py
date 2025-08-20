import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import reference_to_physical_t3, \
    tensor_tensor_leg, tensor_gradtensor_leg


def localstiff(nodes: np.ndarray,
               bounding_box: np.ndarray,
               element_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
               orders: np.ndarray,
               diffusion: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
               advection: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
               reaction: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
               forcing: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]):
    """
    This function calculates the local stiffness matrix associated to the diffusion component of the PDE. The
    requirements on the shapes of the coefficient functions are in the DGFEM class .add_data method.

    Args:
        nodes (np.ndarray): The vertices of the triangle in question
        bounding_box (np.ndarray): The bounding box for the element in question
        element_quadrature_rule (typing.Tuple[np.ndarray, np.ndarray]): The quadrature rule for the triangle in question
        in the format (weights, reference points).
        orders (np.ndarray): The orders of the associated tensor-Legendre polynomials
        diffusion (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The diffusion function
        advection (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The advection function
        reaction (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The reaction function
        forcing (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The forcing function

    Returns:
        np.ndarray: Local stiffness matrix.
    """

    # Unpacking quadrature points and weights
    weights, ref_points = element_quadrature_rule

    # Jacobian calculation -- area of triangle
    B = 0.5 * np.vstack((nodes[1, :] - nodes[0, :], nodes[2, :] - nodes[0, :]))
    De_tri = np.abs(np.linalg.det(B))

    weights = De_tri * weights

    P_Qpoints = reference_to_physical_t3(nodes, ref_points)  # Map reference to physical points

    dim_elem = orders.shape[0]
    z = np.zeros((dim_elem, dim_elem))  # Pre-allocate space to save repeated calculations

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    tensor_leg_array = tensor_tensor_leg(P_Qpoints, m, h, orders)
    gradtensor_leg_array = tensor_gradtensor_leg(P_Qpoints, m, h, orders)

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
                   bounding_box1: np.ndarray, bounding_box2: np.ndarray,
                   edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                   orders: np.ndarray,
                   element_nodes_1: np.ndarray, element_nodes_2: np.ndarray,
                   k_1_area: float, k_2_area: float,
                   polydegree: int,
                   sigma_D: float,
                   normal: np.ndarray,
                   diffusion: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
                   advection: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    """
    This function calculates the local stiffness matrix contributions from a given interior edge of the mesh.

    Args:
        nodes (np.ndarray): The vertices of the edge in question.
        bounding_box1 (np.ndarray): The bounding box for the first element in question.
        bounding_box2 (np.ndarray): The bounding box for the second element in question.
        edge_quadrature_rule (typing.Tuple[np.ndarray, np.ndarray]): The quadrature rule for the edge in question.
        orders (np.ndarray): The orders of the associated tensor-Legendre polynomials.
        element_nodes_1 (np.ndarray): The vertices of the first element in question.
        element_nodes_2 (np.ndarray): The vertices of the second element in question.
        k_1_area (float): The area of the first element in question.
        k_2_area (float): The area of the second element in question.
        polydegree (int): The max degree of Legendre polynomial used (this is used for the interior penalty terms).
        sigma_D (float): The global penalisation parameter.
        normal (np.ndarray): The normal vector of the edge. The direction is pre-determined.
        diffusion (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The diffusion function
        advection (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The advection function

    Returns:
        np.ndarray: The local stiffness matrix corresponding to both elements and both directions.
    """

    # Unpacking quadrature points and weights
    weights, ref_Qpoints = edge_quadrature_rule

    # Convert reference to physical quadrature points and calculate length value, De, and midpoint, mid_point
    mid_point = np.mean(nodes, axis=0, keepdims=True)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = mid_point.repeat(ref_Qpoints.shape[0], axis=0)
    P_Qpoints = ref_Qpoints @ tanvec[:, None].T + C
    De = np.linalg.norm(tanvec)

    dim_elem = orders.shape[0]
    z = np.zeros((2 * dim_elem, 2 * dim_elem))  # pre-allocate space to save repeated calculations

    # Information for two bounding box 1,2 n is the normal vector from k1 to k2
    h1 = 0.5 * np.array([bounding_box1[1] - bounding_box1[0], bounding_box1[3] - bounding_box1[2]])
    m1 = 0.5 * np.array([bounding_box1[1] + bounding_box1[0], bounding_box1[3] + bounding_box1[2]])
    h2 = 0.5 * np.array([bounding_box2[1] - bounding_box2[0], bounding_box2[3] - bounding_box2[2]])
    m2 = 0.5 * np.array([bounding_box2[1] + bounding_box2[0], bounding_box2[3] + bounding_box2[2]])

    tensor_leg_array = np.stack(
        (tensor_tensor_leg(P_Qpoints, m1, h1, orders),
         tensor_tensor_leg(P_Qpoints, m2, h2, orders)), axis=1)

    if diffusion is not None:

        # Penalty term
        lambda_dot = normal @ diffusion(mid_point).squeeze() @ normal

        abs_k_b_1 = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes_1 - nodes[0, :]))))
        abs_k_b_2 = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes_2 - nodes[0, :]))))

        # Assuming p-coverability
        c_inv_1 = min(k_1_area / abs_k_b_1, polydegree ** 2)
        c_inv_2 = min(k_2_area / abs_k_b_2, polydegree ** 2)
        sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * max(c_inv_1 / k_1_area, c_inv_2 / k_2_area)

        # Assuming not p-coverable
        # sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * max(1.0 / abs_k_b_1, 1.0 / abs_k_b_2)

        gradtensor_leg_array = np.stack(
            (tensor_gradtensor_leg(P_Qpoints, m1, h1, orders),
             tensor_gradtensor_leg(P_Qpoints, m2, h2, orders)), axis=1)

        a_val = diffusion(P_Qpoints)

        a_gradx_array = np.einsum('ijk,nij -> nij', a_val, gradtensor_leg_array[:, 0])
        a_grady_array = np.einsum('ijk,nij -> nij', a_val, gradtensor_leg_array[:, 1])

        # Precompute arrays
        agx = a_gradx_array @ normal
        agy = a_grady_array @ normal

        agx_w = agx * weights.T
        agy_w = agy * weights.T

        weighted_tensor = tensor_leg_array * weights.T

        # Compute elementwise stiffness matrix contributions
        auxiliary_sigma_1 = (-sigma * tensor_leg_array[:, 0] @ weighted_tensor[:, 0].T +
                             0.5 * (agx @ weighted_tensor[:, 0].T + tensor_leg_array[:, 0] @ agx_w.T))
        auxiliary_sigma_2 = (sigma * tensor_leg_array[:, 1] @ weighted_tensor[:, 0].T +
                             0.5 * (agy @ weighted_tensor[:, 0].T - tensor_leg_array[:, 1] @ agx_w.T))
        auxiliary_sigma_3 = (sigma * tensor_leg_array[:, 0] @ weighted_tensor[:, 1].T +
                             0.5 * (-agx @ weighted_tensor[:, 1].T + tensor_leg_array[:, 0] @ agy_w.T))
        auxiliary_sigma_4 = (-sigma * tensor_leg_array[:, 1] @ weighted_tensor[:, 1].T +
                             0.5 * (-agy @ weighted_tensor[:, 1].T - tensor_leg_array[:, 1] @ agy_w.T))

        # Sort contributions and assemble
        local_1 = auxiliary_sigma_1
        local_2 = np.triu(auxiliary_sigma_2).T + np.triu(auxiliary_sigma_3, 1)
        local_3 = np.triu(auxiliary_sigma_3).T + np.triu(auxiliary_sigma_2, 1)
        local_4 = auxiliary_sigma_4

        z -= np.vstack((np.hstack((local_1.T, local_3.T)), np.hstack((local_2.T, local_4.T))))

    if advection is not None:

        # Correct the normal vector's direction if required.
        correction = np.sum(advection(mid_point).flatten() * normal) >= 1e-12
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

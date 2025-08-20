import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import (reference_to_physical_t3,
                                                                    tensor_tensor_leg, tensor_gradtensor_leg)


def error_element(
        nodes: np.ndarray,
        bounding_box: np.ndarray,
        element_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
        orders: np.ndarray,
        dg_coefs: np.ndarray,
        u_exact: typing.Callable[[np.ndarray], np.ndarray],
        grad_u_exact: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
        diffusion: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None,
        auxilliary_function: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]] = None
    ) -> (float, float, float):
    """
    This function calculates the various norms required over the elements themselves and returns the 'L2 component' of
    the L2 norm, dG norm and the H1 semi-norm.

    Args:
        nodes (np.ndarray): The vertices of the triangle in question
        bounding_box (np.ndarray): The bounding box for the element in question
        element_quadrature_rule (typing.Tuple[np.ndarray, np.ndarray]): The quadrature rule for the triangle in question
        in the format (weights, reference points).
        orders (np.ndarray): The orders of the associated tensor-Legendre polynomials
        dg_coefs (np.ndarray): The coefficient vector of the solution vector for a given element.
        u_exact (typing.Callable[[np.ndarray], np.ndarray]): The true solution function.
        grad_u_exact (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The true gradient solution function.
        diffusion (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The diffusion function
        auxilliary_function (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The auxilliary function

    Returns:
        (float, float, float): The values of the local L2 and dG norm as well as the optional H1 semi-norm.
    """

    weights, ref_points = element_quadrature_rule
    B = 0.5 * np.vstack((nodes[1, :] - nodes[0, :], nodes[2, :] - nodes[0, :]))
    De_tri = np.abs(np.linalg.det(B))

    weights = De_tri * weights

    # Map reference to physical points
    P_Qpoints = reference_to_physical_t3(nodes, ref_points)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    # Getting function values for the quadrature
    u_val = u_exact(P_Qpoints)
    c0_val = auxilliary_function(P_Qpoints)

    # Getting dG solutions values for the quadrature
    tensor_leg_array = tensor_tensor_leg(P_Qpoints, m, h, orders)
    u_DG_val = tensor_leg_array.T @ dg_coefs

    t2, t4 = 0.0, 0.0

    if diffusion is not None:

        a_val = diffusion(P_Qpoints)
        a_val = a_val.reshape(a_val.shape[0], a_val.shape[1] * a_val.shape[2])
        grad_u_val = grad_u_exact(P_Qpoints)

        gradtensor_leg_array = tensor_gradtensor_leg(P_Qpoints, m, h, orders)

        grad_u_DG = np.vstack((gradtensor_leg_array[:, :, 0].T @ dg_coefs,
                               gradtensor_leg_array[:, :, 1].T @ dg_coefs)).T

        grad = grad_u_val - grad_u_DG
        t4 = np.sum(grad ** 2, axis=1)  # H1 semi-norm values

        grad_11 = grad[:, 0] * grad[:, 0]
        grad_12 = grad[:, 0] * grad[:, 1]
        grad_21 = grad[:, 1] * grad[:, 0]
        grad_22 = grad[:, 1] * grad[:, 1]

        grad = np.hstack((grad_11[:, np.newaxis], grad_12[:, np.newaxis],
                          grad_21[:, np.newaxis], grad_22[:, np.newaxis]))

        t2 = np.sum(grad * a_val, axis=1)  # dG norm values for the diffusion term.

    t1 = (u_val - u_DG_val) ** 2  # L2 norm values
    t3 = c0_val * t1  # dG norm values for the advection reacion terms.

    l2_subnorm: float = 0.5 * np.dot(t1, weights)[0]
    dg_subnorm: float = 0.5 * np.dot(t2 + t3, weights)[0]
    h1_subnorm: float = 0.5 * np.dot(t4, weights)[0]

    if diffusion is None:
        return l2_subnorm, dg_subnorm, None

    return l2_subnorm, dg_subnorm, h1_subnorm


def error_interface(nodes: np.ndarray,
                    bounding_box1: np.ndarray, bounding_box2: np.ndarray,
                    edge_quadrature_rule: typing.Tuple[np.ndarray, np.ndarray],
                    orders: np.ndarray,
                    element_nodes_1: np.ndarray, element_nodes_2: np.ndarray,
                    k_1_area: float, k_2_area: float,
                    polydegree: int,
                    sigma_D: float,
                    dg_coefs1: np.ndarray, dg_coefs2: np.ndarray,
                    normal: np.ndarray,
                    diffusion: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]],
                    advection: typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]) -> float:
    """
        This function calculates the dG sub norm associated with the jump terms on the interior facets.

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
            dg_coefs1 (np.ndarray): The coefficient vector of the solution vector for the first element in question.
            dg_coefs2 (np.ndarray): The coefficient vector of the solution vector for the second element in question.
            normal (np.ndarray): The normal vector of the edge. The direction is pre-determined.
            diffusion (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The diffusion function
            advection (typing.Optional[typing.Callable[[np.ndarray], np.ndarray]]): The advection function

        Returns:
            float: The dg subnorm associated with the jump terms on the interior facets.
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

    # Information for two bounding box 1,2 n is the normal vector from k1 to k2
    h1 = 0.5 * np.array([bounding_box1[1] - bounding_box1[0], bounding_box1[3] - bounding_box1[2]])
    m1 = 0.5 * np.array([bounding_box1[1] + bounding_box1[0], bounding_box1[3] + bounding_box1[2]])
    h2 = 0.5 * np.array([bounding_box2[1] - bounding_box2[0], bounding_box2[3] - bounding_box2[2]])
    m2 = 0.5 * np.array([bounding_box2[1] + bounding_box2[0], bounding_box2[3] + bounding_box2[2]])

    tensor_leg_array = np.stack(
        (tensor_tensor_leg(P_Qpoints, m1, h1, orders),
         tensor_tensor_leg(P_Qpoints, m2, h2, orders)), axis=1)

    u_DG_val1 = np.matmul(tensor_leg_array[:, 0, :].T, dg_coefs1)  # DG solution on kappa1
    u_DG_val2 = np.matmul(tensor_leg_array[:, 1, :].T, dg_coefs2)  # DG solution on kappa2

    t = (u_DG_val1 - u_DG_val2) ** 2  # Jump terms

    dg_subnorm: float = 0.0

    if diffusion is not None:

        # Penalty term
        lambda_dot = normal @ diffusion(mid).squeeze() @ normal

        abs_k_b_1 = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes_1 - nodes[0, :]))))
        abs_k_b_2 = np.max(0.5 * np.abs(abs(np.cross(nodes[1, :] - nodes[0, :], element_nodes_2 - nodes[0, :]))))

        # Assuming p-coverability
        c_inv_1 = min(k_1_area / abs_k_b_1, polydegree ** 2)
        c_inv_2 = min(k_2_area / abs_k_b_2, polydegree ** 2)
        sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * max(c_inv_1 / k_1_area, c_inv_2 / k_2_area)

        # Assuming not p-coverable
        # sigma = sigma_D * lambda_dot * polydegree ** 2 * (2 * De) * max(1.0 / abs_k_b_1, 1.0 / abs_k_b_2)

        dg_subnorm += sigma * np.dot(t, weights)[0]

    if advection is not None:
        b_val = advection(P_Qpoints)
        dg_subnorm += 0.5 * np.dot(np.abs(np.sum(b_val * normal[None, :], axis=1)) * t, weights)[0]

    return dg_subnorm

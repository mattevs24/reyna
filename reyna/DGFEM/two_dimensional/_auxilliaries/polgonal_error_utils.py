import typing

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.assembly import assembly_aux


def a_grad_norm(nodes: np.ndarray,
                bounding_box: np.ndarray,
                dg_coefs: np.ndarray,
                Po: int,
                Lege_ind: np.ndarray,
                grad_u_exact: typing.Callable[[np.ndarray], np.ndarray],
                diffusion: typing.Callable[[np.ndarray], np.ndarray]) -> (float, float):

    """
    This function calculates two (semi-) norms over a given simplex; the sub-DG norm, sqrt(a)grad(u-u_h), as well as
    the H1 seminorm respectively.

    Args:
        nodes: The nodes of the simplex in question.
        bounding_box: The bounding box of the element which contains the simplex.
        dg_coefs: The DG coefficients corresponding to the element in question.
        Po: The precision of the quadrature required.
        Lege_ind: The indecies of the tensored Legendre polynomials required (must match with dg_coefs).
        grad_u_exact: The gradient of the true solution to the PDE.
        diffusion: The diffusion operator of the PDE.

    Returns:
        (float, float): The DG sub-norm and H1 semi-norm over the simplex in question.
    """

    dim_elem = Lege_ind.shape[0]

    quadrature_order = 0.5 * np.ceil((Po + 3))
    w_x, x = assembly_aux.quad_GL(quadrature_order)
    w_y, y = assembly_aux.quad_GJ1(quadrature_order)

    quad_x = np.kron(x, np.ones((w_y.shape[0], 1)))
    quad_y = np.kron(np.ones((w_x.shape[0], 1)), y)
    weights = np.kron(w_x, w_y)

    shiftpoints = np.hstack((0.5 * (1.0 + quad_x) * (1.0 - quad_y) - 1.0, quad_y))
    ref_points = 0.5 * shiftpoints + 0.5

    B = np.vstack((nodes[1, :] - nodes[0, :], nodes[2, :] - nodes[0, :]))
    De_tri = np.abs(np.linalg.det(B))
    De = 0.25 * De_tri

    P_Qpoints = assembly_aux.reference_to_physical_t3(nodes, ref_points)

    a_val = diffusion(P_Qpoints)
    a_val = a_val.reshape(a_val.shape[0], a_val.shape[1] * a_val.shape[2])
    grad_u_val = grad_u_exact(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    # construct the matrix for all the local basis funcitons
    Px = np.zeros((P_Qpoints.shape[0], dim_elem))
    Py = np.zeros((P_Qpoints.shape[0], dim_elem))

    for i in range(dim_elem):
        t = assembly_aux.gradtensor_leg(P_Qpoints, m, h, Lege_ind[i, :])
        Px[:, i] = t[:, 0]
        Py[:, i] = t[:, 1]

    grad_u_DG = np.vstack((np.matmul(Px, dg_coefs), np.matmul(Py, dg_coefs))).T  # gradient of DG

    t2 = 0.5 * np.sum((grad_u_val - grad_u_DG) ** 2, axis=1)
    h1_subnorm = De * np.dot(t2.T, weights)[0]

    grad1 = grad_u_val - grad_u_DG
    grad2 = grad_u_val - grad_u_DG

    grad_11 = grad1[:, 0] * grad2[:, 0]
    grad_12 = grad1[:, 0] * grad2[:, 1]
    grad_21 = grad1[:, 1] * grad2[:, 0]
    grad_22 = grad1[:, 1] * grad2[:, 1]

    grad = np.hstack((grad_11[:, np.newaxis], grad_12[:, np.newaxis],
                      grad_21[:, np.newaxis], grad_22[:, np.newaxis]))

    t3 = 0.5 * np.sum(grad * a_val, axis=1)

    dg_subnorm = De * np.dot(t3, weights)

    return dg_subnorm, h1_subnorm


def cr_err_elem(nodes: np.ndarray,
                bounding_box: np.ndarray,
                dg_coefs: np.ndarray,
                Po: int,
                Lege_ind: np.ndarray,
                u_exact: typing.Callable[[np.ndarray], np.ndarray],
                auxiliiary_function: typing.Callable[[np.ndarray], np.ndarray]) -> (float, float):

    """
    This function calculates two norms over a simplex: the DG sub-norm, c_0(u-u_h) and the L2 norm.

    Args:
        nodes: The nodes of the simplex in question.
        bounding_box: The bounding box of the element which contains the simplex.
        dg_coefs: The DG coefficients corresponding to the element in question.
        Po: The precision of the quadrature required.
        Lege_ind: The indecies of the tensored Legendre polynomials required (must match with dg_coefs).
        u_exact: The true solution to the PDE.
        auxiliiary_function: The auxilliary operator of the PDE: c_0 = c - 0.5 * div(b).

    Returns:
        (float, float): The DG sub-norm and L2-norm over the simplex in question.
    """

    dim_elem = Lege_ind.shape[0]

    w_x, x = assembly_aux.quad_GL(np.ceil((Po + 1) * 0.5))
    w_y, y = assembly_aux.quad_GJ1(np.ceil((Po + 1) * 0.5))

    quad_x = np.kron(x, np.ones((w_y.shape[0], 1)))
    quad_y = np.kron(np.ones((w_x.shape[0], 1)), y)
    weights = np.kron(w_x, w_y)

    shiftpoints = np.hstack(((1 + quad_x) * (1 - quad_y) * 0.5 - 1, quad_y))
    ref_points = 0.5 * shiftpoints + 0.5

    B = np.vstack((nodes[1, :] - nodes[0, :], nodes[2, :] - nodes[0, :]))
    De_tri = np.abs(np.linalg.det(B))
    De = 0.25 * De_tri

    P_Qpoints = assembly_aux.reference_to_physical_t3(nodes, ref_points)

    # data for quadrature
    u_val = u_exact(P_Qpoints)
    c0_val = auxiliiary_function(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    # construct the matrix for all the local basis function
    P = np.zeros((P_Qpoints.shape[0], dim_elem))

    for i in range(dim_elem):
        P[:, i] = assembly_aux.tensor_leg(P_Qpoints, m, h, Lege_ind[i, :])

    u_DG_val = np.matmul(P, dg_coefs)  # DG solution

    t1 = 0.5 * (u_val - u_DG_val) ** 2
    t2 = c0_val * t1

    dg_subnorm = De * np.dot(t2, weights)
    l2_subnorm = De * np.dot(t1, weights)

    return dg_subnorm, l2_subnorm


def err_interface(nodes: np.ndarray,
                  vertices: np.ndarray,
                  bounding_box1: np.ndarray,
                  bounding_box2: np.ndarray,
                  dg_coefs1: np.ndarray,
                  dg_coefs2: np.ndarray,
                  normal: np.ndarray,
                  Po: int,
                  Lege_ind: np.ndarray,
                  diffusion: typing.Callable[[np.ndarray], np.ndarray],
                  sigma: float) -> float:
    """
    This function calculates the DG sub-norm, sigma [[u-u_h]]**2, over an interior facet.

    Args:
        nodes: The nodes of the simplex in question.
        vertices: The vertices of the simplices which contain this interior edge.
        bounding_box1: The bounding box of one element that contains the edge.
        bounding_box2: The bounding box of the other element that contains the edge.
        dg_coefs1: The DG coefficients corresponding to the first element in question.
        dg_coefs2: The DG coefficients corresponding to the second element in question.
        normal: The normal vector to the edge in question.
        Po: The precision of the quadrature required.
        Lege_ind: The indecies of the tensored Legendre polynomials required (must match with dg_coefs).
        diffusion: The diffusion operator of the PDE.
        sigma: The diffusion penalty parameter.

    Returns:
        (float): The DG sub-norm over the edge in question.

    """

    h1 = 0.5 * np.array([bounding_box1[1] - bounding_box1[0], bounding_box1[3] - bounding_box1[2]])
    m1 = 0.5 * np.array([bounding_box1[1] + bounding_box1[0], bounding_box1[3] + bounding_box1[2]])
    h2 = 0.5 * np.array([bounding_box2[1] - bounding_box2[0], bounding_box2[3] - bounding_box2[2]])
    m2 = 0.5 * np.array([bounding_box2[1] + bounding_box2[0], bounding_box2[3] + bounding_box2[2]])

    dim_elem = Lege_ind.shape[0]  # number of basis for each element

    # generating quadrature points and weights

    weights, ref_Qpoints = assembly_aux.quad_GL(np.ceil((Po + 1) * .5))

    # change the quadrature nodes from reference domain to physical domain.

    mid = 0.5 * np.sum(nodes, axis=0)
    mid = mid[np.newaxis, :]
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = np.kron(mid, np.ones((ref_Qpoints.shape[0], 1)))
    P_Qpoints = np.kron(ref_Qpoints, tanvec) + C
    De = np.linalg.norm(nodes[1, :] - nodes[0, :]) * 0.5

    # penalty term
    lambda_dot = np.einsum('i,ij,j->', normal, diffusion(mid[np.newaxis, :]).squeeze(), normal)
    to_be_summed = (vertices - np.kron(nodes[0, :],
                                       np.ones((vertices.shape[0], 1)))) * np.kron(normal,
                                                                                   np.ones((vertices.shape[0], 1)))
    measure_B = De * np.max(np.abs(np.sum(to_be_summed, axis=1)))
    sigma = 2 * sigma * lambda_dot * De / measure_B

    # construct the matrix for all the local basis function

    P1 = np.zeros((P_Qpoints.shape[0], dim_elem))
    P2 = np.zeros((P_Qpoints.shape[0], dim_elem))

    for i in range(dim_elem):
        P1[:, i] = assembly_aux.tensor_leg(P_Qpoints, m1, h1, Lege_ind[i, :])
        P2[:, i] = assembly_aux.tensor_leg(P_Qpoints, m2, h2, Lege_ind[i, :])

    u_DG_val1 = np.matmul(P1, dg_coefs1)  # DG solution at kappa1
    u_DG_val2 = np.matmul(P2, dg_coefs2)  # DG solution at kappa2

    # Part 2 DG norm error
    # sigma*jump{u_DG}^2+2a(grad_u - aver{grad_U_DG})\cdot jump{u_DG}

    t = sigma * (u_DG_val1 - u_DG_val2) ** 2
    DGPart_2 = De * np.dot(t.T, weights)

    return DGPart_2


def cr_err_interface(nodes: np.ndarray,
                     bounding_box1: np.ndarray,
                     bounding_box2: np.ndarray,
                     dg_coefs1: np.ndarray,
                     dg_coefs2: np.ndarray,
                     normal: np.ndarray,
                     Po: int,
                     Lege_ind: np.ndarray,
                     advection: typing.Callable[[np.ndarray], np.ndarray]) -> float:
    """
    This function calculates the DG sub-norm, (u-u_h)^+ - (u-u_h)^-, over an interior facet. This is the

    Args:
        nodes: The nodes of the simplex in question.
        bounding_box1: The bounding box of one element that contains the edge.
        bounding_box2: The bounding box of the other element that contains the edge.
        dg_coefs1: The DG coefficients corresponding to the first element in question.
        dg_coefs2: The DG coefficients corresponding to the second element in question.
        normal: The normal vector to the edge in question.
        Po: The precision of the quadrature required.
        Lege_ind: The indecies of the tensored Legendre polynomials required (must match with dg_coefs).
        advection: The advection component of the PDE.

    Returns:
        (float): The DG sub-norm over the edge in question.
    """

    h1 = 0.5 * np.array([bounding_box1[1] - bounding_box1[0], bounding_box1[3] - bounding_box1[2]])
    m1 = 0.5 * np.array([bounding_box1[1] + bounding_box1[0], bounding_box1[3] + bounding_box1[2]])
    h2 = 0.5 * np.array([bounding_box2[1] - bounding_box2[0], bounding_box2[3] - bounding_box2[2]])
    m2 = 0.5 * np.array([bounding_box2[1] + bounding_box2[0], bounding_box2[3] + bounding_box2[2]])

    dim_elem = Lege_ind.shape[0]  # number of basis for each element

    weights, ref_Qpoints = assembly_aux.quad_GL(np.ceil((Po + 1) * 0.5))

    # change the quadrature nodes from reference domain to physical domain.
    mid = 0.5 * np.sum(nodes, axis=0)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = np.kron(mid, np.ones((ref_Qpoints.shape[0], 1)))
    P_Qpoints = np.kron(ref_Qpoints, tanvec) + C
    De = np.linalg.norm(nodes[1, :] - nodes[0, :]) * 0.5

    # data for quadrature, function value b and normal vector n_vec
    b_val = advection(P_Qpoints)
    n_vec = np.kron(normal, np.ones((ref_Qpoints.shape[0], 1)))

    # construct the matrix for all the local basis function
    P1 = np.zeros((P_Qpoints.shape[0], dim_elem))
    P2 = np.zeros((P_Qpoints.shape[0], dim_elem))

    for i in range(dim_elem):
        P1[:, i] = assembly_aux.tensor_leg(P_Qpoints, m1, h1, Lege_ind[i, :])
        P2[:, i] = assembly_aux.tensor_leg(P_Qpoints, m2, h2, Lege_ind[i, :])

    u_DG_val1 = np.matmul(P1, dg_coefs1)  # DG solution on kappa1
    u_DG_val2 = np.matmul(P2, dg_coefs2)  # DG solution on kappa2

    # Part 2 DG norm error  int_\e  1/2*|b \cdot n |*[u_DG]^2 dx

    t = 0.5 * np.abs(np.sum(b_val * n_vec, axis=1)) * (u_DG_val1 - u_DG_val2) ** 2
    DGPart_2 = De * np.dot(t, weights)

    return DGPart_2


def err_bd_face(nodes: np.ndarray,
                vertices: np.ndarray,
                bounding_box: np.ndarray,
                df_coefs: np.ndarray,
                normal: np.ndarray,
                Po: int,
                Lege_ind: np.ndarray,
                u_exact: typing.Callable[[np.ndarray], np.ndarray],
                diffusion: typing.Callable[[np.ndarray], np.ndarray],
                sigma: float) -> float:

    """
    This function calculates the DG sub-norm, sigma [[u-u_h]]**2, over a boundary facet.

    Args:
        nodes: The nodes of the simplex in question.
        vertices: The vertices of the simplices which contain this boundary edge.
        bounding_box: The bounding box of the element which contains the simplex.
        df_coefs: The DG coefficients corresponding to the element in question.
        normal: The OPUNV to the boundary facet.
        Po: The precision of the quadrature required.
        Lege_ind: The indecies of the tensored Legendre polynomials required (must match with dg_coefs).
        u_exact: The true solution to the PDE.
        diffusion: The diffusion component to the PDE.
        sigma: The diffusion penalty parameter.

    Returns:
        (float): The DG sub-norm over the boundary facet in question.
    """

    dim_elem = Lege_ind.shape[0]

    weights, ref_Qpoints = assembly_aux.quad_GL(np.ceil((Po + 1) * 0.5))

    # change the quadrature nodes from reference domain to physical domain.
    mid = 0.5 * np.sum(nodes, axis=0)
    mid = mid[np.newaxis, :]
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = np.kron(mid, np.ones((ref_Qpoints.shape[0], 1)))
    P_Qpoints = np.kron(ref_Qpoints, tanvec) + C
    De = 0.5 * np.linalg.norm(nodes[1, :] - nodes[0, :])

    # penalty term
    lambda_dot = np.einsum('i,ij,j->', normal, diffusion(mid[np.newaxis, :]).squeeze(), normal)
    to_be_summed = (vertices - np.kron(nodes[0, :],
                                       np.ones((vertices.shape[0], 1)))) * np.kron(normal,
                                                                                   np.ones((vertices.shape[0], 1)))
    measure_B = De * np.max(np.abs(np.sum(to_be_summed, axis=1)))
    sigma = 2 * sigma * lambda_dot * De / measure_B
    # data for quadrature, function value b and normal vector n_vec
    u_val = u_exact(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    P = np.zeros((P_Qpoints.shape[0], dim_elem))

    for i in range(dim_elem):
        P[:, i] = assembly_aux.tensor_leg(P_Qpoints, m, h, Lege_ind[i, :])

    u_DG_val = np.matmul(P, df_coefs)  # DG solution

    # Prat 3 DG norm error
    t = sigma * (u_val - u_DG_val) ** 2
    DGPart_3 = De * np.dot(t.T, weights)

    return DGPart_3


def cr_err_bd_face(nodes: np.ndarray,
                   bounding_box: np.ndarray,
                   df_coefs: np.ndarray,
                   normal: np.ndarray,
                   Po: int,
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
        Po: The precision of the quadrature required.
        Lege_ind: The indecies of the tensored Legendre polynomials required (must match with dg_coefs).
        u_exact: The true solution to the PDE.
        advection: The advection component to the PDE.

    Returns:
        (float): The DG sub-norm over the boundary facet in question.
    """

    dim_elem = Lege_ind.shape[0]  # number of basis for each element

    weights, ref_Qpoints = assembly_aux.quad_GL(np.ceil((Po + 1) * 0.5))

    mid = 0.5 * np.sum(nodes, axis=0)
    tanvec = 0.5 * (nodes[1, :] - nodes[0, :])
    C = np.kron(mid, np.ones((ref_Qpoints.shape[0], 1)))
    P_Qpoints = np.kron(ref_Qpoints, tanvec) + C
    De = 0.5 * np.linalg.norm(nodes[1, :] - nodes[0, :])

    # data for quadrature, function value b and normal vector n_vec
    b_val = advection(P_Qpoints)
    n_vec = np.kron(normal, np.ones((ref_Qpoints.shape[0], 1)))
    u_val = u_exact(P_Qpoints)

    h = 0.5 * np.array([bounding_box[1] - bounding_box[0], bounding_box[3] - bounding_box[2]])
    m = 0.5 * np.array([bounding_box[1] + bounding_box[0], bounding_box[3] + bounding_box[2]])

    P = np.zeros((P_Qpoints.shape[0], dim_elem))

    for i in range(dim_elem):
        P[:, i] = assembly_aux.tensor_leg(P_Qpoints, m, h, Lege_ind[i, :])

    u_DG_val = np.matmul(P, df_coefs)  # DG solution

    # Part 3 DG norm error  int_\e  1/2*|b \cdot n |*(u-u_DG)^2 dx
    t = 0.5 * np.abs(np.sum(b_val * n_vec, axis=1)) * (u_val - u_DG_val) ** 2
    DGPart_3 = De * np.dot(t, weights)

    return DGPart_3

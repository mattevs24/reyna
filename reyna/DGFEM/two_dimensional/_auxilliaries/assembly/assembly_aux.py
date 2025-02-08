import importlib_resources as resources

import numpy as np

from reyna.DGFEM.two_dimensional._auxilliaries.polygonal_basis_utils import LegendreP


def quad_GL(n: int):

    file_path = str(resources.files('reyna._data.quadratures').joinpath(f"array_GL_{int(n)}"))
    data = np.atleast_2d(np.loadtxt(file_path, delimiter=","))
    ref_points = data[:, :1]
    weights = data[:, 1:]

    return weights, ref_points


def quad_GJ1(n: int):

    file_path = str(resources.files('reyna._data.quadratures').joinpath(f"array_GJ1_{int(n)}"))
    data = np.atleast_2d(np.loadtxt(file_path, delimiter=","))
    ref_points = data[:, :1]
    weights = data[:, 1:]

    return weights, ref_points


def reference_to_physical_t3(t: np.ndarray, ref: np.ndarray):

    phy = np.dot(np.column_stack([1.0 - ref[:, 0] - ref[:, 1], ref[:, 0], ref[:, 1]]), t)

    return phy


def shift_leg_derivative(x, m, h, order, k):
    tol = np.finfo(float).eps
    y = (x - m) / h

    mask = np.abs(y) > 1.0
    y[mask] = (1.0 - tol) * np.sign(y[mask])

    if order <= k - 1:
        val_p = np.zeros(y.shape[0])
    else:
        correction = 1.0 if k == 0 else np.sqrt((order + 1.0) * order)
        P = LegendreP(y, k, order - k) * correction

        val_p = h ** (-0.5 - k) * P

    return val_p


def tensor_leg(x, m, h, order):
    val = shift_leg_derivative(x[:, 0], m[0], h[0], order[0], 0) * \
          shift_leg_derivative(x[:, 1], m[1], h[1], order[1], 0)
    return val


def gradtensor_leg(x, m, h, order):

    val = np.zeros((x.shape[0], 2))
    
    shift_leg_der_11 = shift_leg_derivative(x[:, 0], m[0], h[0], order[0], 1)
    shift_leg_der_12 = shift_leg_derivative(x[:, 1], m[1], h[1], order[1], 0)
    shift_leg_der_21 = shift_leg_derivative(x[:, 0], m[0], h[0], order[0], 0)
    shift_leg_der_22 = shift_leg_derivative(x[:, 1], m[1], h[1], order[1], 1)
    
    val[:, 0] = shift_leg_der_11 * shift_leg_der_12
    val[:, 1] = shift_leg_der_21 * shift_leg_der_22

    return val


def generate_local_quadrature(simplex_nodes: np.ndarray,
                              quadrature_precision: int) -> (np.ndarray, np.ndarray):
    # quadrature data
    quadrature_order = int(np.ceil(0.5 * (quadrature_precision + 1)))
    w_x, x = quad_GL(quadrature_order)
    w_y, y = quad_GJ1(quadrature_order)

    quad_x = np.reshape(np.repeat(x, w_y.shape[0]), (-1, 1))
    quad_y = np.reshape(np.tile(y, w_x.shape[0]), (-1, 1), order='F')
    weights = (w_x[:, None] * w_y).flatten().reshape(-1, 1)

    # The duffy points and the reference triangle points.
    shiftpoints = np.hstack((0.5 * (1.0 + quad_x) * (1.0 - quad_y) - 1.0, quad_y))
    ref_points = 0.5 * shiftpoints + 0.5

    # Jacobian calculation
    B = 0.5 * np.vstack((simplex_nodes[1, :] - simplex_nodes[0, :], simplex_nodes[2, :] - simplex_nodes[0, :]))
    De_tri = np.abs(np.linalg.det(B))

    # The physical points
    P_Qpoints = reference_to_physical_t3(simplex_nodes, ref_points)

    return De_tri * weights, P_Qpoints

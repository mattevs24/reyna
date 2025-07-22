import importlib_resources as resources

import numpy as np
from numba import njit, float64, int64

from reyna.DGFEM.two_dimensional._auxilliaries.polygonal_basis_utils import tensor_LegendreP


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


@njit(float64[:, :](float64[:], float64, float64, int64, float64[:]))
def tensor_shift_leg(x, _m, _h, polydegree, correction = np.array([np.nan])):
    tol = 2.220446049250313e-16
    y = (x - _m) / _h

    mask = np.abs(y) > 1.0
    y[mask] = (1.0 - tol) * np.sign(y[mask])
    if np.isnan(correction[0]):
        P = _h ** (-0.5) * tensor_LegendreP(y, 0, polydegree)
        return P
    else:
        P = _h ** (-1.5) * tensor_LegendreP(y, 1, polydegree - 1) * np.expand_dims(correction, axis=1)
        new_P = np.empty((P.shape[0] + 1, P.shape[1]))
        new_P[0, :] = 0.0
        new_P[1:, :] = P
        return new_P


@njit(float64[:, :](float64[:, :], float64[:], float64[:], int64[:, :]))
def tensor_tensor_leg(x, _m, _h, orders):
    polydegree = np.max(orders)
    val = tensor_shift_leg(x[:, 0], _m[0], _h[0], polydegree, correction = np.array([np.nan]))[orders[:, 0], :] * \
        tensor_shift_leg(x[:, 1], _m[1], _h[1], polydegree, correction = np.array([np.nan]))[orders[:, 1], :]

    return val


@njit(float64[:, :, :](float64[:, :], float64[:], float64[:], int64[:, :]))
def tensor_gradtensor_leg(x, _m, _h, orders):

    val = np.zeros((orders.shape[0], x.shape[0], 2))
    polydegree = np.max(orders)
    correction = np.array([np.sqrt((i + 1.0) * i) for i in range(1, polydegree + 1)])

    shift_leg_der_11 = tensor_shift_leg(x[:, 0], _m[0], _h[0], polydegree, correction)[orders[:, 0], :]

    shift_leg_der_12 = tensor_shift_leg(x[:, 1], _m[1], _h[1], polydegree, correction = np.array([np.nan]))[orders[:, 1], :]
    shift_leg_der_21 = tensor_shift_leg(x[:, 0], _m[0], _h[0], polydegree, correction = np.array([np.nan]))[orders[:, 0], :]

    shift_leg_der_22 = tensor_shift_leg(x[:, 1], _m[1], _h[1], polydegree, correction)[orders[:, 1], :]

    val[..., 0] = shift_leg_der_11 * shift_leg_der_12
    val[..., 1] = shift_leg_der_21 * shift_leg_der_22

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

import numpy as np
from numba import njit, f8, i8

from reyna.DGFEM.two_dimensional._auxilliaries.polygonal_basis_utils import tensor_LegendreP


@njit(f8[:, :](f8[:, :], f8[:, :]))
def reference_to_physical_t3(t: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    This function takes a set of vertices of a triangle and a reference set of quadrature points and maps these points
    to the triangle.
    Args:
        t (np.ndarray): The vertices of the triangle in an array with shape (3, 2).
        ref (np.ndarray): The reference quadrature points in an array with shape (N, 2) where N varies with the
        precision of the quadrature rule in question.
    Returns:
        np.ndarray: The mapped reference points in an array of shape (N, 2).
    """

    n = ref.shape[0]

    arr = np.zeros((n, 3), dtype=np.float64)
    arr[:, 0] = 1.0 - ref[:, 0] - ref[:, 1]
    arr[:, 1] = ref[:, 0]
    arr[:, 2] = ref[:, 1]

    phy = np.dot(arr, np.ascontiguousarray(t))

    return phy


@njit(f8[:, :](f8[:], f8, f8, i8, f8[:]))
def tensor_shift_leg(x: np.ndarray, _m: float, _h: float, polydegree: int, correction=np.array([np.nan])) -> np.ndarray:
    """
    This function generates the legendre polynomial values at a given set of 1D input points. This is coupled with
    the midpoint and half-extent values (along a given dimension) to tailor the function to the bounding box in
    question. The correction term is added to calculate the gradient terms.

    Args:
        x (np.ndarray): The quadrature points for a given element/facet.
        _m (float): The midpoint of the corresponding element.
        _h (float): The half-extents of the corresponding element
        polydegree (int): The highest total polynomial power.
        correction (np.ndarray): The corerction term for differentiation.

    Returns:
        np.ndarray: A numpy array containing the tensor legendre polynomial values.
    """

    tol = 2.220446049250313e-16
    y = (x - _m) / _h  # Recentre points

    mask = np.abs(y) > 1.0
    y[mask] = (1.0 - tol) * np.sign(y[mask])  # Reset points to within the reference element -- precaution

    if np.isnan(correction[0]):
        # No derivatives required
        P = _h ** (-0.5) * tensor_LegendreP(y, 0, polydegree)
        return P
    else:
        # Derivatives required -- correction required.
        P = _h ** (-1.5) * tensor_LegendreP(y, 1, polydegree - 1) * np.expand_dims(correction, axis=1)
        new_P = np.empty((P.shape[0] + 1, P.shape[1]))
        new_P[0, :] = 0.0
        new_P[1:, :] = P
        return new_P


@njit(f8[:, :](f8[:, :], f8[:], f8[:], i8[:, :]))
def tensor_tensor_leg(x: np.ndarray, _m: np.ndarray, _h: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """
    This function generates the values for the tensor-legendre polynomials. It takes the values from each cartesian
    dimension and multiplies. This is a tensor function and vectorises the point-wise calculations.

    Args:
        x (np.ndarray): The points in which the tensor-lengendre polynomials are evaluated of shape (M, 2)
        _m (np.ndarray): The midpoint of the cartesian bounding box for the element.
        _h (np.ndarray): The half-extent of the cartesian bounding box for the element.
        orders (np.ndarray): The orders of the tensor-lengendre polynomials for each direction: needs to be an integer
        array of shape (N, 2). For orders[:, 0], the corresponding tensor-lengendre polynomial is
        L_{orders[0, 0]}(x)L_{orders[0, 1]}(y).

    Returns:
        np.ndarray: The tensor-lengendre polynomial values at the given points. This will be of the shape (N, M).
    """

    polydegree = np.max(orders)
    val = tensor_shift_leg(x[:, 0], _m[0], _h[0], polydegree, correction=np.array([np.nan]))[orders[:, 0], :] * \
        tensor_shift_leg(x[:, 1], _m[1], _h[1], polydegree, correction=np.array([np.nan]))[orders[:, 1], :]

    return val


@njit(f8[:, :, :](f8[:, :], f8[:], f8[:], i8[:, :]))
def tensor_gradtensor_leg(x: np.ndarray, _m: np.ndarray, _h: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """
    Thie function takes a set of input points and returns the evaluated gradients of the tensor-lengendre polynomials.
    Args:
        x (np.ndarray): The points in which the tensor-lengendre polynomials are evaluated of shape (M, 2)
        _m (np.ndarray): The midpoint of the cartesian bounding box for the element.
        _h (np.ndarray): The half-extent of the cartesian bounding box for the element.
        orders (np.ndarray): The orders of the tensor-lengendre polynomials for each direction: needs to be an integer
        array of shape (N, 2). For orders[:, 0], the corresponding gradient tensor-lengendre polynomial is
        [L_{orders[0, 0]}'(x)L_{orders[0, 1]}(y), L_{orders[0, 0]}(x)L_{orders[0, 1]}'(y)].

    Returns:
        np.ndarray: The tensor-lengendre polynomial values at the given points. This will be of the shape (N, M, 2).
    """

    val = np.zeros((orders.shape[0], x.shape[0], 2))
    polydegree = np.max(orders)

    # Correction term for the gradient operators
    correction = np.array([np.sqrt((i + 1.0) * i) for i in range(1, polydegree + 1)])

    shift_leg_der_11 = tensor_shift_leg(x[:, 0], _m[0], _h[0], polydegree, correction)[orders[:, 0], :]

    shift_leg_der_12 = tensor_shift_leg(
        x[:, 1], _m[1], _h[1], polydegree,
        correction=np.array([np.nan])
    )[orders[:, 1], :]

    shift_leg_der_21 = tensor_shift_leg(
        x[:, 0], _m[0], _h[0], polydegree,
        correction=np.array([np.nan])
    )[orders[:, 0], :]

    shift_leg_der_22 = tensor_shift_leg(x[:, 1], _m[1], _h[1], polydegree, correction)[orders[:, 1], :]

    val[..., 0] = shift_leg_der_11 * shift_leg_der_12
    val[..., 1] = shift_leg_der_21 * shift_leg_der_22

    return val

import numpy as np
from numba import njit, f8, i8

from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import tensor_shift_leg


@njit(f8[:, :](f8[:, :], f8[:, :]))
def reference_to_physical_t4(t: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    This function takes a set of vertices of a simplex and a reference set of quadrature points and maps these points
    to the simplex.
    Args:
        t (np.ndarray): The vertices of the simplex in an array with shape (4, 3).
        ref (np.ndarray): The reference quadrature points in an array with shape (N, 3) where N varies with the
        precision of the quadrature rule in question.
    Returns:
        np.ndarray: The mapped reference points in an array of shape (N, 3).
    """

    # TODO: need to sort this function here but first need to taget the quadrature rule?
    # TODO: need to sort this too for facet reference to physical points also

    n = ref.shape[0]

    arr = np.zeros((n, 3), dtype=np.float64)
    arr[:, 0] = 1.0 - ref[:, 0] - ref[:, 1]
    arr[:, 1] = ref[:, 0]
    arr[:, 2] = ref[:, 1]

    phy = np.dot(arr, np.ascontiguousarray(t))

    return phy


@njit(f8[:, :](f8[:, :], f8[:], f8[:], i8[:, :]))
def tensor_tensor_leg(x: np.ndarray, _m: np.ndarray, _h: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """
    This function generates the values for the tensor-legendre polynomials. It takes the values from each cartesian
    dimension and multiplies. This is a tensor function and vectorises the point-wise calculations.

    Args:
        x (np.ndarray): The points in which the tensor-lengendre polynomials are evaluated of shape (M, d)
        _m (np.ndarray): The midpoint of the cartesian bounding box for the element (shape (d,)).
        _h (np.ndarray): The half-extent of the cartesian bounding box for the element (shape (d,)).
        orders (np.ndarray): The orders of the tensor-lengendre polynomials for each direction: needs to be an integer
        array of shape (N, d). For orders[:, 0], the corresponding tensor-lengendre polynomial is
        L_{orders[0, 0]}(x_1)*...*L_{orders[0, d-1]}(x_d).

    Returns:
        np.ndarray: The tensor-lengendre polynomial values at the given points. This will be of the shape (N, M).
    """

    # TODO: I think this is now generalised into all dimensions >=1?

    polydegree = np.max(orders)
    dimension = len(_m)  # or x.shape[1] or len(_h) etc....

    constructor = lambda j: tensor_shift_leg(x[:, j], _m[j], _h[j], polydegree, correction=np.array([np.nan]))
    val = np.prod([constructor(i)[orders[:, i], :] for i in range(dimension)], axis=0)

    return val


@njit(f8[:, :, :](f8[:, :], f8[:], f8[:], i8[:, :]))
def tensor_gradtensor_leg(x: np.ndarray, _m: np.ndarray, _h: np.ndarray, orders: np.ndarray) -> np.ndarray:
    """
    Thie function takes a set of input points and returns the evaluated gradients of the tensor-lengendre polynomials.

    Args:
        x (np.ndarray): The points in which the tensor-lengendre polynomials are evaluated of shape (M, d)
        _m (np.ndarray): The midpoint of the cartesian bounding box for the element (shape (d,)).
        _h (np.ndarray): The half-extent of the cartesian bounding box for the element (shape (d,)).
        orders (np.ndarray): The orders of the tensor-lengendre polynomials for each direction: needs to be an integer
        array of shape (N, d).
    Returns:
        np.ndarray: The tensor-lengendre polynomial values at the given points. This will be of the shape (N, M, d).
    """

    # TODO: I think this should work for all dimensions >=1? need to optimise the code in this file for sure, are
    #  lambda constructors the best here?

    # Initial parameters
    polydegree = np.max(orders)
    dimension = len(_m)

    # Correction term for the gradient operators
    correction = np.array([np.sqrt((i + 1.0) * i) for i in range(1, polydegree + 1)])

    shift_leg_constructor = lambda j: tensor_shift_leg(x[:, j], _m[j], _h[j], polydegree, correction=np.array([np.nan]))
    shift_leg = [shift_leg_constructor(i)[orders[:, i], :] for i in range(dimension)]

    shift_leg_der_constructor = lambda j: tensor_shift_leg(x[:, j], _m[j], _h[j], polydegree, correction)
    shift_leg_der = [shift_leg_der_constructor(i)[orders[:, i], :] for i in range(dimension)]

    val = np.zeros((orders.shape[0], *x.shape))

    for i in range(dimension):
        val[..., i] = np.prod(shift_leg[:i]) * shift_leg_der[i] * np.prod(shift_leg[i+1:])

    return val

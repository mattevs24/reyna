from math import gamma

import numpy as np
from numba import njit, f8, i8, intc


def Basis_index2D(polydegree: int) -> np.ndarray:
    """
    Returns the FEM basis indices for a given polynomial degree and dimension.

    Args:
        polydegree: (int) Degree of the polynomial approximation space.
    Returns:
        (np.ndarray) FEM basis indices.
    Raises:
        ValueError: If polydegree is not an positive integer (>=0).
    """
    if polydegree < 0:
        raise ValueError('Input "polydegree" must be non-negative (>=0).')

    t_indices = np.indices((polydegree + 1, polydegree + 1)).reshape(2, -1).T

    mask_index = np.sum(t_indices, axis=1) <= polydegree
    FEM_index = t_indices[mask_index]

    return FEM_index


@njit(f8[:, :](f8[:], intc, i8))
def tensor_LegendreP(x: np.ndarray, power: int, max_order: int) -> np.ndarray:
    """
    This function generates the Legendre polynomial basis values up to a given order N. This function can also calculate
    the derivative values, given by the power value.
    Args:
        x (np.ndarray): The set of points which the Legendre polynomials will be evaluated.
        power (int): The number of derivatives required.
        max_order (int): The maximum order to which the polynomial evaluations are made.

    Returns:
        np.ndarray: The Legendre polynomial basis values.
    """

    n_points = x.shape[0]
    init_array = np.empty((max_order + 1, n_points), dtype=np.float64)

    # Compute gamma0 using math.gamma (Numba supports it)
    gamma0 = (2.0 ** (2.0 * power + 1.0) / (2.0 * power + 1.0) *
              (gamma(power + 1.0) ** 2) / gamma(2.0 * power + 1.0))

    c0 = 1.0 / np.sqrt(gamma0)
    for i in range(n_points):
        init_array[0, i] = c0

    if max_order == 0:
        return init_array

    gamma1 = ((power + 1.0) ** 2 / (2.0 * power + 3.0)) * gamma0
    c1 = (power + 1.0) / np.sqrt(gamma1)
    for i in range(n_points):
        init_array[1, i] = c1 * x[i]

    if max_order == 1:
        return init_array

    # Precompute aold
    aold = 1.0 / (power + 1.0) * np.sqrt((power + 1.0) ** 2 / (2.0 * power + 3.0))

    for j in range(1, max_order):
        h1 = 2.0 * (j + power)
        anew = 1.0 / (j + power + 1.0) * np.sqrt(
            (j + 1.0) * (j + 1.0 + 2.0 * power) * (j + 1.0 + power) ** 2 / ((h1 + 1.0) * (h1 + 3.0))
        )

        init_array[j + 1, :] = (x * init_array[j, :] - aold * init_array[j - 1, :]) / anew

        aold = anew

    return init_array

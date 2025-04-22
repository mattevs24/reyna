from math import gamma

import numpy as np


def Basis_index2D(polydegree: int) -> np.ndarray:
    """
    Returns the FEM basis indices for a given polynomial degree and dimension.

    Args:
        polydegree: (int) Degree of the polynomial approximation space.
    Returns:
        (np.ndarray) FEM basis indices.
    Raises:
        None
    """
    if polydegree < 0:
        raise ValueError('Input "polydegree" must be non-negative (>=0).')

    t_indices = np.indices((polydegree + 1, polydegree + 1)).reshape(2, -1).T

    mask_index = np.sum(t_indices, axis=1) <= polydegree
    FEM_index = t_indices[mask_index]

    return FEM_index


def tensor_LegendreP(x: np.ndarray, power: int, N: int) -> np.ndarray:

    init_array = np.zeros((N + 1, x.shape[0]), dtype=float)

    # Initial values P_0(x) and P_1(x)
    gamma0 = 2 ** (2.0 * power + 1.0) / (2.0 * power + 1.0) * gamma(power + 1) ** 2 / gamma(2.0 * power + 1)
    init_array[0, :] = 1.0 / np.sqrt(gamma0) * np.ones(x.shape)

    if N == 0:
        return init_array

    gamma1 = (power + 1.0) * (power + 1.0) / (2.0 * power + 3.0) * gamma0
    init_array[1, :] = ((power + 1.0) * x) / np.sqrt(gamma1)

    if N == 1:
        return init_array

    # Precompute aold for recurrence relation
    aold = 1.0 / (1.0 + power) * np.sqrt((power + 1) * (power + 1) / (2 * power + 3))

    for j in range(1, N):
        h1 = 2 * (j + power)
        anew = 1 / (j + power + 1) * np.sqrt(
            (j + 1) * (j + 1 + 2 * power) * (j + 1 + power) * (j + 1 + power) / (h1 + 1) / (h1 + 3)
        )
        init_array[j+1, :] = (x * init_array[j, :] - aold * init_array[j-1,:]) / anew
        aold = anew

    return init_array
import numpy as np


def basis_indices(polydegree: int, dimension: int) -> np.ndarray:
    """
    Returns the FEM basis indices for a given polynomial degree and dimension.

    Args:
        polydegree (int): Degree of the polynomial approximation space.
        dimension (int): Dimension of the computational domain.
    Returns:
        (np.ndarray) Legendre basis indices.
    Raises:
        ValueError: If polydegree is not an positive integer (>=0).
    """
    if polydegree < 0:
        raise ValueError('Input "polydegree" must be non-negative (>=0).')

    t_indices = np.indices([polydegree + 1] * dimension).reshape(dimension, -1).T

    mask_index = np.sum(t_indices, axis=1) <= polydegree
    orders = t_indices[mask_index]

    return orders

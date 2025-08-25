import numpy as np
import typing
import matplotlib.pyplot as plt


def d_sphere(p: np.ndarray, center: typing.Optional[np.ndarray] = None, radius: float = 1.0) -> np.ndarray:
    """Calculates signed distances from points to a spherical domain.

    Given an array of coordinates `p` in the form `[p_0, p_1, ..., p_n]` where each `p_i` is `[x_i, y_i]` or
    `[x_i, y_i, z_i]`, this function computes the distance of each point from the center of a sphere. The distances are
    "translated" by the radius such that they are negative inside the sphere and positive outside.

    Args:
        p (np.ndarray): Array of points of the form `[p_0, p_1, ..., p_n]` where each `p_i` is `[x_i, y_i]` or
        `[x_i, y_i, z_i]`.
        center (np.ndarray, optional): The center of the spherical domain. Defaults to the origin of the corresponding
        dimension if `None`.
        radius (float, optional): Radius of the spherical domain. Defaults to `1.0`.

    Returns:
        np.ndarray: Array of signed distances in the form `[d_0, d_1, ..., d_n]`,
        where negative values indicate points inside the sphere and positive values outside.

    Raises:
        Exception: p must contain points of dimention 2 or 3.
        Exception: center must be of equal dimensio to p.
    """

    if center is None:
        center = np.zeros((1, p.shape[1]))

    p_shape = p.shape[1]
    center_shape = center.shape
    if not 2 <= p_shape <= 3:
        raise Exception(f"p must contain points of dimention 2 or 3: p has points of dimension {p_shape}")

    if center_shape == (p_shape,):
        center = center[:, np.newaxis].T
    elif not center_shape == (1, p_shape):
        raise Exception(f"center must be a points of equal to those in p: center has points of dimension"
                        f" {center_shape}: must be either ({p_shape},) or (1, {p_shape})")

    d = (np.sqrt(np.sum((p - center) ** 2, axis=1)) - radius)[:, np.newaxis]
    d = np.concatenate((d, d), axis=1)

    return d


def d_rectangle(p: np.ndarray, x1: float, x2: float, y1: float, y2: float) -> np.ndarray:
    """Calculates signed distances from points to a rectangular domain.

    Given an array of 2D coordinates `p`, this function computes the signed distance of to the rectangular domain.

    Args:
        p (np.ndarray): Array of 2D points.
        x1 (float): Lower x bound.
        x2 (float): Upper x bound.
        y1 (float): Lower y bound.
        y2 (float): Upper y bound.

    Returns:
        np.ndarray: Array of signed distances.
    """
    d = [x1 - p[:, 0], p[:, 0] - x2, y1 - p[:, 1], p[:, 1] - y2]
    d = np.vstack(d)
    max_d = np.max(d, axis=0)
    d = np.vstack((d, max_d)).T
    return d


def d_line(p: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    """Calculates signed distances from points to a line.

        Given an array of 2D coordinates `p`, this function computes the signed distance to a given line. This is a
        constructor object and can be used to build more complicated domains. Defined with two points which the line
        passes through. The interior side is defined as the positive normal.

        Args:
            p (np.ndarray): Array of 2D points.
            x1 (float): First points x value.
            y1 (float): First points y value.
            x2 (float): Second points x value.
            y2 (float): Second points y value.

        Returns:
            np.ndarray: Array of signed distances.
        """
    # tangent vector
    a = np.array([x2 - x1, y2 - y1])
    a = a/np.linalg.norm(a)

    # re-center points around the origin
    p[:, 0] = p[:, 0] - x1
    p[:, 1] = p[:, 1] - y1

    # test the dot product with the normal
    d = (p[:, 0] * a[1] - p[:, 1] * a[0])[:, np.newaxis]
    d = np.concatenate((d, d), axis=1)
    return d


def d_hexagon(p: np.ndarray, x: float = 0.0, y: float = 0.0, scale: float = 1.0) -> np.ndarray:
    vertices = np.array([
        [scale * np.cos(i * np.pi / 3), scale * np.sin(i * np.pi / 3)] for i in range(6)
    ])

    p[:, 0] -= x
    p[:, 1] -= y

    d_0 = d_line(p, *vertices[0, :], *vertices[1, :])
    d_1 = d_line(p, *vertices[1, :], *vertices[2, :])
    d_2 = d_line(p, *vertices[2, :], *vertices[3, :]) - scale * 0.5 * np.sqrt(3.0)
    d_3 = d_line(p, *vertices[3, :], *vertices[4, :]) - scale * np.sqrt(3.0)
    d_4 = d_line(p, *vertices[4, :], *vertices[5, :]) - scale * np.sqrt(3.0)
    d_5 = d_line(p, *vertices[5, :], *vertices[0, :]) - scale * 0.5 * np.sqrt(3.0)

    d_01 = d_intersect(d_0, d_1)
    d_012 = d_intersect(d_01, d_2)
    d_0123 = d_intersect(d_012, d_3)
    d_01234 = d_intersect(d_0123, d_4)
    d = d_intersect(d_01234, d_5)

    return d


def d_intersect(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:

    d = _d_combination(d1, d2, "max")
    return d


def d_union(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:

    d = _d_combination(d1, d2, "min")
    return d


def d_difference(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:

    d = _d_combination(d1, d2, "diff")
    return d


def _d_combination(d1: np.ndarray, d2: np.ndarray, functionality: str) -> np.ndarray:
    """Combines two distance arrays using a specified operation on their last column.

        This function concatenates the non-last columns of `d1` and `d2` and then applies an operation on the last
        column of each array according to the `functionality` argument. The resulting array includes all original
        columns plus the combined last column.

        This is used to construct the more complicated 'd_intersect', 'd_union', and 'd_difference' functions.

        Args:
            d1 (np.ndarray): First array of distances with shape (n, m1), where the last column is used for the
            combination operation.
            d2 (np.ndarray): Second array of distances with shape (n, m2), where the last column is used for the
            combination operation.
            functionality (str): Specifies how to combine the last columns of `d1` and `d2`. Supported options are:
            - "min": Take the element-wise minimum. - "max": Take the element-wise maximum. - "diff": Take the
            element-wise maximum of `d1` and the negative of `d2`.

        Returns:
            np.ndarray: Combined array containing the original columns from `d1` and `d2`
            (excluding their original last columns) and the new last column resulting
            from the specified combination.

        Raises:
            ValueError: If `functionality` is not one of "min", "max", or "diff".
        """

    if functionality == "min":
        func_d = np.min(np.concatenate((d1[:, -1][:, np.newaxis], d2[:, -1][:, np.newaxis]), axis=1), axis=1)[:,
                 np.newaxis]
    elif functionality == "max":
        func_d = np.max(np.concatenate((d1[:, -1][:, np.newaxis], d2[:, -1][:, np.newaxis]), axis=1), axis=1)[:,
                 np.newaxis]
    elif functionality == "diff":
        func_d = np.max(np.concatenate((d1[:, -1][:, np.newaxis], -d2[:, -1][:, np.newaxis]), axis=1), axis=1)[:,
                 np.newaxis]
    else:
        raise ValueError("The input for functionality is invalid: poly_mesher_distance_functionc._d_combination")

    d1 = d1[:, :d1.shape[1] - 1]
    d2 = d2[:, :d2.shape[1] - 1]
    if len(d1.shape) == 1 or len(d2.shape) == 1:
        d = np.concatenate((d1[:, np.newaxis], d2[:, np.newaxis]), axis=1)
    else:
        d = np.concatenate((d1, d2), axis=1)

    d = np.concatenate((d, func_d), axis=1)
    return d

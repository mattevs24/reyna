import numpy as np


def d_cuboid(p: np.ndarray, bounding_box: np.ndarray) -> np.ndarray:
    """
    Inputs an array of points of the form [[x0, y0, z0], ..., [xn, yn, zn]] and outputs the distances to each side of
    the corresponding cube defined by the bounding box given by the input `bounding_box`.

    Args:
        p (np.ndarray): An array of 3D points to calculate the distances to the boundaries from.
        bounding_box (np.ndarray): An array of 3D boundaries to calculate the distances to the boundaries from.

    Returns:
        np.ndarray: An array of the distances.
    """
    d = np.array(
        [bounding_box[0, 0] - p[:, 0], p[:, 0] - bounding_box[1, 0],
         bounding_box[0, 1] - p[:, 1], p[:, 1] - bounding_box[1, 1],
         bounding_box[0, 2] - p[:, 2], p[:, 2] - bounding_box[1, 2]]
    )

    d = np.concatenate((d, np.max(d)))
    return d

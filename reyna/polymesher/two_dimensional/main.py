"""
This is the mesh algorithm used for the polygonal meshing. Credits for the orginal code goes to Talischi et al. [1]
written in Matlab. Several modifications are in place here for efficiency as well as functionality. For example, the
cleaning function in the original code has several bugs including a tendency to collapse boundary edges. In terms of
more recent work (see Calloo et al. [2]), we require Voronoi tessellations, a property of which is not retained by the
cleaning functions. There is the potential for small edges but either more or less iterations fixes this.

[1] Talischi, C., Paulino, G. H., Pereira, A., & Menezes, I. F. PolyMesher: a general-purpose mesh generator for
polygonal elements written in Matlab. Structural and Multidisciplinary Optimization. (2012)

[2] Calloo, A., Evans, M., Lockyer, H., Madiot, F., Pryer, T., & Zanetti, L. Cycle-Free Polytopal Mesh Sweeping for
Boltzmann Transport. arXiv preprint arXiv:2412.01660. (2024)
"""

import itertools
import time

import numpy as np

from scipy.spatial import Voronoi
from shapely import Polygon

from reyna.polymesher.two_dimensional._auxilliaries.abstraction import PolyMesh, Domain


def poly_mesher(domain: Domain, max_iterations: int = 100, **kwargs) -> PolyMesh:
    """
    The PolyMesher function. This function generates a bounded Voronoi tesselation of a given domain. The user must
    initialise the function with either the number of elements they require by submitting the 'n_points' keyword
    argument or initialise with a grid of points by inputting the 'n_xy' keyword argument. There is a further verbose
    argument also.

    Args:
        domain (Domain): This is a 'Domain' object, be that custom or a pre-defined one.
        max_iterations (int): This is the number of iterations required by the user. Generally, the more iterations, the
        more uniform the elements are in shape and number of edges etc.
        **kwargs: There are several valid inputs here. 'verbose' has obvious effects and is a boolean value (default
        False). 'cleaned' cleans up the resulting mesh and removes small numerical artifacts. 'seed' sets a seed for the
        generation of the initial points. There are two options for the initial points. One can either input 'n_points'
        to dictate a given number of points or 'n_xy' (of type (int, int)) together to generate a Cartesian mesh.

    Returns:
        (PolyMesh): A PolyMesh object containing all the relevant information to be used in a geometry function.

    Raises:
        AttributeError: If the keyword 'n_points' or 'n_xy' are missed.

    Notes:
        - If the 'cleaned' keyword is used, the output mesh is not guarenteed to be a Voronoi diagram.

    See Also:
        [PolyMesher](https://paulino.princeton.edu/conferences/presentations/11periera_polymesher.pdf)

    """

    verbose: bool = False
    if 'verbose' in kwargs:
        verbose = kwargs.pop('verbose')

    cleaned_mesh: bool = False
    if 'cleaned' in kwargs:
        cleaned_mesh = kwargs.pop('cleaned')

    points = _poly_mesher_init_point_set(domain, **kwargs)

    fixed_points = domain.pFix()  # from here can call domain.fixed_points -- this initialises the property simult
    if fixed_points is not None:
        points = np.concatenate((fixed_points, points), axis=0)
        n_fixed = fixed_points.shape[0]
    else:
        n_fixed = 0

    iteration, error, tolerance = 0, 1.0, 1e-4
    bounding_box = domain.bounding_box

    area = (bounding_box[0, 1] - bounding_box[0, 0]) * (bounding_box[1, 1] - bounding_box[1, 0])

    while iteration <= max_iterations and error > tolerance:

        _time = time.time()
        reflected_points = _poly_mesher_reflect(points, domain, area)

        voronoi = Voronoi(np.concatenate((points, reflected_points), axis=0), qhull_options='Qbb Qz')

        empty_ind = voronoi.regions.index([])
        sorting = np.argsort(voronoi.point_region)
        del voronoi.regions[empty_ind]
        elements = [np.array(voronoi.regions[i]) for i in np.argsort(sorting)]

        if iteration > max_iterations - 1 or error <= 2.0 * tolerance:
            vertices, regions = _poly_mesher_extract_nodes(voronoi.vertices, elements[: points.shape[0]])

            if cleaned_mesh:
                vertices, regions = _poly_mesher_extract_nodes(vertices, regions)
                vertices, regions = _poly_mesher_collapse_small_edges(vertices, regions, 0.1)

                return PolyMesh(vertices, regions, points, domain)

            return PolyMesh(vertices, regions, points, domain)

        points, area, error = _poly_mesher_vorocentroid(points, voronoi.vertices, elements)

        if fixed_points is not None:
            points[:n_fixed, :] = fixed_points

        iteration += 1

        if iteration % 10 == 0 and verbose:
            print(f"Iteration: {iteration}. Error: {error}")


def _poly_mesher_init_point_set(domain: Domain, **kwargs) -> np.ndarray:
    """
    The function which defines the initial point set for the poly_mesher function.
    Args:
        domain (Domain): The Domain object associated with the computational domain.
        **kwargs: Keyword arguments. There are several valid inputs here.

    Returns:
        np.ndarray: The initial point set for the poly_mesher function.

    Raises:
        AttributeError: If the keyword 'n_points' or 'n_xy' are missed.
    """

    bounding_box = domain.bounding_box

    if 'seed' in kwargs:
        np.random.seed(kwargs.pop('seed'))

    if 'n_points' in kwargs:
        # This generates a random point set
        n_points = kwargs.get('n_points')
        points = np.full((n_points, 2), -np.inf)
        s = 0
        while s < n_points:
            p_1 = (bounding_box[0, 1] - bounding_box[0, 0]) * \
                np.random.uniform(size=(1, n_points)).T + bounding_box[0, 0]
            p_2 = (bounding_box[1, 1] - bounding_box[1, 0]) * \
                np.random.uniform(size=(1, n_points)).T + bounding_box[1, 0]

            p = np.concatenate((p_1, p_2), axis=1)
            d = domain.distances(p)
            last_index_negative = np.argwhere(d[:, -1] < 0.0)
            number_added = min(n_points - s, last_index_negative.shape[0])
            points[s:s + number_added, :] = p[last_index_negative[:number_added].T.flatten(), :]
            s += number_added

    elif 'n_xy' in kwargs:
        # This generates a uniformly spread point set
        n_x, n_y = kwargs.get('n_xy')

        x = np.linspace(bounding_box[0, 0], bounding_box[0, 1], n_x + 1)
        y = np.linspace(bounding_box[1, 0], bounding_box[1, 1], n_y + 1)
        x_c = 0.5 * (x[1:] + x[:-1])
        y_c = 0.5 * (y[1:] + y[:-1])
        [X, Y] = np.meshgrid(x_c, y_c)

        X, Y = X.T, Y.T
        points = np.concatenate((np.reshape(X, (-1, 1), order='F'), np.reshape(Y, (-1, 1), order="F")), axis=1)
        d = domain.distances(points)
        log_ind = d[:, -1] < 0.0
        points = points[log_ind, :]

    else:
        raise AttributeError("key word error: must be just `n_points` or both `n_x` and `n_y`.")

    return points


def _poly_mesher_reflect(points: np.ndarray, domain: Domain, area: float) -> np.ndarray:
    """
    This function reflects points near the boundary of the domain to generate a clean boundary.

    Args:
        points (np.ndarray): The set of points to consider reflecting.
        domain (Domain): The Domain object associated with the computational domain.
        area (float): The (approximated) area of the domain.

    Returns:
        np.ndarray: The reflected points.
    """

    epsilon = 1.0e-8
    n_points = points.shape[0]
    alpha = 1.5 * np.sqrt(area / float(n_points))

    d = domain.distances(points)
    n_boundary_segments = d.shape[1] - 1

    eps_array = np.array([epsilon, 0.0])
    n_1 = 1.0 / epsilon * (domain.distances(points + eps_array) - d)

    eps_array = np.array([0.0, epsilon])
    n_2 = 1.0 / epsilon * (domain.distances(points + eps_array) - d)

    log_ind = np.abs(d[:, :n_boundary_segments]) < alpha

    p_1 = np.tile(points[:, 0][:, np.newaxis], (1, n_boundary_segments))
    p_2 = np.tile(points[:, 1][:, np.newaxis], (1, n_boundary_segments))

    p_1 = p_1.T[log_ind.T][:, np.newaxis]
    p_2 = p_2.T[log_ind.T][:, np.newaxis]

    n_1 = n_1[:, :-1].T[log_ind.T][:, np.newaxis]
    n_2 = n_2[:, :-1].T[log_ind.T][:, np.newaxis]

    d = d[:, :-1].T[log_ind.T][:, np.newaxis]

    r_ps = np.concatenate((p_1, p_2), axis=1) - 2.0 * np.concatenate((n_1, n_2), axis=1) * np.tile(d, (1, 2))

    r_p_ds = domain.distances(r_ps)

    logical_rp = np.logical_and(r_p_ds[:, -1] > 0, np.abs(r_p_ds[:, -1]) >= 0.9 * np.abs(d).flatten())

    r_ps = r_ps[logical_rp]

    if not r_ps.size == 0:
        r_ps = np.unique(r_ps, axis=0)

    return r_ps


def _poly_mesher_vorocentroid(points: np.ndarray, vertices, elements) -> (np.ndarray, float, float):
    """
    This function computes several related and important features. The element centroids, the numerical total area as
    well as the error value associated wih non-centroidal Voronoi diagrams.

    Args:
        points: The set of current Voronoi centres.
        vertices: The vertices of the current Voronoi diagram.
        elements: The elements of the current Voronoi diagram.

    Returns:
        (np.ndarray, float, float): An array of the centroid of the current Voronoi diagram, the numerical total area
        (i.e. the sum of the areas of the Voronoi elements -- generally not equal to the area of the computational
        domain) and the error value associated wih non-centroidal Voronoi diagrams.
    """

    n_points = points.shape[0]
    center_points = np.full((n_points, 2), -np.inf)
    areas = np.full((n_points,), -np.inf)

    for i in range(n_points):
        region = Polygon(vertices[elements[i]])
        areas[i] = region.area
        center_points[i, :] = np.array(region.centroid.coords.xy).T.flatten()

    total_area = areas.sum()
    error = np.sqrt(np.sum((areas ** 2) * np.sum((center_points - points) ** 2, 1), 0)) * n_points / total_area ** 1.5

    return center_points, total_area, error


def _poly_mesher_extract_nodes(nodes: np.ndarray, filtered_elements: list) -> (np.ndarray, list):
    """
    This function extracts the vertices and elements of the Voronoi diagram which are unique.
    Args:
        nodes: The vertices of the Voronoi diagram.
        filtered_elements: The elements of the Voronoi diagram.

    Returns:
        (np.ndarray, list): Reconstructed nodes and elements which are suitable for numerical methods.

    """

    linked_elements = np.array(list((itertools.chain.from_iterable(filtered_elements))))
    unique_ = np.unique(linked_elements)  # these are the points solely used by the elements
    c_nodes = np.arange(nodes.shape[0])
    c_nodes[list(set(c_nodes).difference(set(unique_)))] = max(unique_)
    nodes, elements = _poly_mesher_rebuild_lists(nodes, filtered_elements, c_nodes)

    return nodes, elements


def _poly_mesher_rebuild_lists(nodes: np.ndarray, elements: list, c_nodes: np.ndarray) -> (np.ndarray, list):
    """
    Thie function removes the copied nodes and rebuilts the elements and node sets.
    Args:
        nodes: The vertices of the Voronoi diagram.
        elements: The elements of the Voronoi diagram.
        c_nodes: The array indicating the operation required for the node in question.

    Returns:
        (np.ndarray, list): Reconstructed nodes and elements without the duplicate vertices.
    """

    _elems = []
    _, idx, r_idx = np.unique(c_nodes, return_index=True, return_inverse=True)

    # idx are the indices of the unique elements -- ie the unique nodes

    if nodes.shape[0] > len(idx):
        idx[-1] = np.max(c_nodes)

    nodes = nodes[idx, :]

    for elem in elements:
        temp_elem = np.unique(r_idx[elem])
        v_ = nodes[temp_elem]
        n_v = len(temp_elem)

        perm = np.argsort(np.arctan2(v_[:, 1] - np.sum(v_[:, 1])/n_v,
                                     v_[:, 0] - np.sum(v_[:, 0])/n_v))

        _elems.append(temp_elem[perm])

    return nodes, _elems


def _poly_mesher_collapse_small_edges(nodes: np.ndarray, elements: list, tolerance: float) -> (np.ndarray, list):
    """
    This function deals with small numerical artifacts of LLoyds algorithm. This method is used when 'cleaned' is used
    as a keyword in the poly_mesher function.

    Args:
        nodes: The vertices of the Voronoi diagram.
        elements: The elements of the Voronoi diagram.
        tolerance: The tolerance parameter associated with the removal of small edges.

    Returns:
        (np.ndarray, list): Cleaned nodes and elements without small edges.
    """

    while True:

        c_edge = np.array([[]])
        for element in elements:
            n_v = len(element)

            if n_v <= 3:
                continue  # this is the case when the element is a triangle -- can't collapse

            v_ = nodes[element]
            beta = np.arctan2(v_[:, 1] - np.sum(v_[:, 1])/n_v, v_[:, 0] - np.sum(v_[:, 0])/n_v)
            beta = np.mod(np.roll(beta, -1) - beta, 2 * np.pi)
            beta_ideal = 2 * np.pi / n_v
            edges = np.concatenate((np.array(element)[:, np.newaxis], np.roll(element, -1)[:, np.newaxis]), axis=1)
            idx = beta < tolerance * beta_ideal
            if (e_ := edges[idx, :]).size != 0:
                if c_edge.size == 0:
                    c_edge = np.atleast_2d(e_)
                else:
                    c_edge = np.concatenate((c_edge, np.atleast_2d(e_)), axis=0)

        if c_edge.size == 0:
            break

        c_edge = np.unique(np.sort(c_edge), axis=0)
        c_nodes = np.arange(nodes.shape[0])
        c_nodes[c_edge[:, 1]] = c_nodes[c_edge[:, 0]]

        nodes, elements = _poly_mesher_rebuild_lists(nodes, elements, c_nodes)

    return nodes, elements

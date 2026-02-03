import typing

import numpy as np
from scipy.spatial import Voronoi, Delaunay

from reyna.polymesher.three_dimensional._auxilliaries.abstraction import Domain3D, PolyMesh3D


def poly_mesher_3d(domain: Domain3D, max_iterations: int = 100, **kwargs) -> PolyMesh3D:
    """

    :param domain:
    :param max_iterations:
    :param kwargs:
    :return:
    """
    if "n_points" in kwargs:
        n_points = kwargs.get("n_points")
        points = poly_mesher_init_point_set(domain, n_points=n_points)

    elif "n_x" in kwargs and "n_y" in kwargs and "n_z" in kwargs:
        n_x = kwargs.get("n_x")
        n_y = kwargs.get("n_y")
        n_z = kwargs.get("n_z")
        points = poly_mesher_init_point_set(domain, n_x=n_x, n_y=n_y, n_z=n_z)

    else:
        raise AttributeError("key word error: must be just `n_points` or both `n_x` and `n_y`.")

    fixed_points = domain.pFix()  # from here can call domain.fixed_points -- this initialises the property simult
    if fixed_points is not None:
        points = np.concatenate((fixed_points, points), axis=0)
        n_fixed = fixed_points.shape[0]
    else:
        n_fixed = 0

    iteration, error, tolerance = 0, 1.0, 1e-4
    area = domain.volume()
    n_points = points.shape[0]

    voronoi, nodes, elements = None, None, None

    while iteration <= max_iterations and error > tolerance:
        reflected_points = poly_mesher_reflect(points, domain, area)

        voronoi = Voronoi(np.concatenate((points, reflected_points), axis=0), qhull_options='Qbb Qz')

        cond_len = len(voronoi.regions[0]) == 0

        nodes = voronoi.vertices
        if cond_len:
            sorting = [np.where(voronoi.point_region == x)[0][0] for x in range(1, len(voronoi.regions))]
            elements = [x for _, x in sorted(zip(sorting, voronoi.regions[1:]))]
        else:
            empty_ind = voronoi.regions.index([])
            sorting_1 = [np.where(voronoi.point_region == x)[0][0] for x in range(0, empty_ind)]
            sorting_2 = [np.where(voronoi.point_region == x)[0][0] for x in range(empty_ind+1, len(voronoi.regions))]
            sorting = sorting_1 + sorting_2
            regions = voronoi.regions
            del regions[empty_ind]
            elements = [x for _, x in sorted(zip(sorting, regions))]

        points, area, error = poly_mesher_vorocentroid(points, nodes, elements)

        if fixed_points is not None:
            points[:n_fixed, :] = fixed_points

        iteration += 1
        # show_mesh(voronoi.vertices, elements[: n_points], bounding_box=bounding_box, alpha=0.4, edgecolor="#556575")

        if iteration % 10 == 0:
            print(f"Iteration: {iteration}. Error: {error}")

    facet_types = np.sum(voronoi.ridge_points < n_points, axis=1)
    print(facet_types)
    mask = facet_types > 0
    filtered_facets = [voronoi.ridge_vertices[i] for i in range(len(voronoi.ridge_vertices)) if mask[i]]
    # filtered_facets = list(filter(lambda x: mask[voronoi.ridge_vertices.index(x)], voronoi.ridge_vertices))

    poly_mesh = PolyMesh3D(
        voronoi.vertices, filtered_facets, elements[: n_points],
        facet_types[mask] - 1, voronoi.ridge_points[mask],
        points, domain
    )
    # TODO: need to be careful -- can use ridge_points and ridge_vertices but have to restrict in some way to include
    #  the boundary facets too -- if a row in ridge points contains an index less than n_points, retain and filter
    #  ridge vertices with this (the indecies should be the global ones?) this could optimise the boundary edge and
    #  interior edge filtering later in DGFEMGeometry?

    return poly_mesh


def poly_mesher_init_point_set(domain: Domain3D, **kwargs) -> np.ndarray:
    """
    This function initialises the point set to be able to use the lloyds algorithm. Can be done uniformly via ```n_x```
    and ```n_y``` or the number of points that reside in the domain via ```n_points```.
    :param domain: Insert the domain of choice here. Must be subclass of the poly_mesher_domain.Domain class.
    :param kwargs: Input either ```n_points``` for number of points within the domain or ```n_x``` and ```n_y``` for
    uniformly spread points throughout the domain (and outside the domain, but these points are simply removed).
    :return: A np.ndarray of points that are within the domain and can be used to intialise the lloyds algorithm.
    """
    # Done: tested fully!
    bounding_box = domain.bounding_box

    if "n_points" in kwargs:
        # This generates a random point set
        n_points = kwargs.get("n_points")
        points = np.full((n_points, 3), -np.inf)
        s = 0
        # np.random.seed(1337)
        while s < n_points:
            p_1 = (bounding_box[0, 1] - bounding_box[0, 0]) * np.random.uniform(size=(1, n_points)).T + bounding_box[0,
                                                                                                                     0]
            p_2 = (bounding_box[1, 1] - bounding_box[1, 0]) * np.random.uniform(size=(1, n_points)).T + bounding_box[1,
                                                                                                                     0]
            p_3 = (bounding_box[2, 1] - bounding_box[2, 0]) * np.random.uniform(size=(1, n_points)).T + bounding_box[2,
                                                                                                                     0]
            p = np.concatenate((p_1, p_2, p_3), axis=1)
            d = domain.distances(p)
            last_index_negative = np.argwhere(d[:, -1] < 0.0)  # index of the seeds within the domain
            number_added = min(n_points - s, last_index_negative.shape[0])
            points[s:s + number_added, :] = p[last_index_negative[:number_added].T.flatten(), :]
            s += number_added

    elif "n_x" in kwargs and "n_y" in kwargs and "n_z" in kwargs:
        # This generates a uniformly spread point set
        n_x = kwargs.get("n_x")
        n_y = kwargs.get("n_y")
        n_z = kwargs.get("n_y")

        x = np.linspace(bounding_box[0, 0], bounding_box[0, 1], n_x + 1)
        y = np.linspace(bounding_box[1, 0], bounding_box[1, 1], n_y + 1)
        z = np.linspace(bounding_box[2, 0], bounding_box[2, 1], n_z + 1)
        x_c = 0.5 * (x[1:] + x[:-1])
        y_c = 0.5 * (y[1:] + y[:-1])
        z_c = 0.5 * (z[1:] + z[:-1])
        [X, Y, Z] = np.meshgrid(x_c, y_c, z_c, indexing="ij")

        X, Y, Z = X.T, Y.T, Z.T
        points = np.concatenate((np.reshape(X, (-1, 1), order='F'),
                                 np.reshape(Y, (-1, 1), order="F"),
                                 np.reshape(Z, (-1, 1), order="F")), axis=1)
        d = domain.distances(points)
        log_ind = d[:, -1] < 0.0
        points = points[log_ind, :]

    else:
        raise AttributeError("key word error: must be just `n_points` or both `n_x` and `n_y`.")

    return points


def poly_mesher_reflect(points: np.ndarray, domain: Domain3D, area: float) -> np.ndarray:
    # done
    """
    Compute the reflection point sets
    :param points:
    :param domain:
    :param area:
    :return:
    """
    print(area)
    epsilon = 1.0e-8
    n_points = points.shape[0]
    alpha = 1.5 * np.sqrt(area / float(n_points))

    d = domain.distances(points)
    n_boundary_segments = d.shape[1] - 1

    n_1 = 1.0 / epsilon * (domain.distances(points + np.tile(np.array([epsilon, 0.0, 0.0]), (n_points, 1))) - d)
    n_2 = 1.0 / epsilon * (domain.distances(points + np.tile(np.array([0.0, epsilon, 0.0]), (n_points, 1))) - d)
    n_3 = 1.0 / epsilon * (domain.distances(points + np.tile(np.array([0.0, 0.0, epsilon]), (n_points, 1))) - d)

    # singles out the points that are within (1.5x) average side length of a region to the boundary
    log_ind = np.abs(d[:, :n_boundary_segments]) < alpha

    p_1 = np.tile(points[:, 0][:, np.newaxis], (1, n_boundary_segments))
    p_2 = np.tile(points[:, 1][:, np.newaxis], (1, n_boundary_segments))
    p_3 = np.tile(points[:, 2][:, np.newaxis], (1, n_boundary_segments))

    p_1 = np.concatenate([p_1[log_ind[:, i], i] for i in range(n_boundary_segments)], axis=0)[:, np.newaxis]
    p_2 = np.concatenate([p_2[log_ind[:, i], i] for i in range(n_boundary_segments)], axis=0)[:, np.newaxis]
    p_3 = np.concatenate([p_3[log_ind[:, i], i] for i in range(n_boundary_segments)], axis=0)[:, np.newaxis]

    n_1 = np.concatenate([n_1[log_ind[:, i], i] for i in range(n_boundary_segments)], axis=0)[:, np.newaxis]
    n_2 = np.concatenate([n_2[log_ind[:, i], i] for i in range(n_boundary_segments)], axis=0)[:, np.newaxis]
    n_3 = np.concatenate([n_3[log_ind[:, i], i] for i in range(n_boundary_segments)], axis=0)[:, np.newaxis]

    d = np.concatenate([d[log_ind[:, i], i] for i in range(n_boundary_segments)], axis=0)[:, np.newaxis]

    r_ps = np.concatenate((p_1, p_2, p_3), axis=1) - 2.0 * np.concatenate((n_1, n_2, n_3), axis=1) * np.tile(d, (1, 3))

    r_p_ds = domain.distances(r_ps)

    logical_rp = np.logical_and(r_p_ds[:, -1] > 0, np.abs(r_p_ds[:, -1]) >= 0.9 * np.abs(d).flatten())

    r_ps = r_ps[logical_rp, :]

    if not r_ps.size == 0:
        # this may be better as an approx equals?
        r_ps = np.unique(r_ps, axis=0)

    return r_ps


def poly_mesher_vorocentroid(points: np.ndarray, vertices, elements) -> (np.ndarray, float, float):
    # [P, node, elem] -> [Pc,Area,Err]
    """
    This function calculates the centroid of a Voronoi cell
    :param points: A set of points as inputs
    :param vertices: The list of vertices of the whole Voronoi diagram
    :param elements: The list of lists of vertex indicies that make up each Voronoi cell
    :return:
    """
    n_points = points.shape[0]
    center_points = np.full((n_points, 3), -np.inf)
    volumes = np.full((n_points,), -np.inf)

    for i in range(n_points):
        center_points[i, :], volumes[i] = _nd_centroid_volume(vertices[elements[i], :])

    total_area = volumes.sum()
    error = np.sqrt(np.sum((volumes ** 2) * np.sum((center_points - points) ** 2, 1), 0)) * n_points / total_area ** 1.5

    return center_points, total_area, error


def _nd_centroid_volume(vertices: np.ndarray) -> (np.ndarray, float):
    simplicialation = Delaunay(vertices)
    n_simp = simplicialation.simplices.shape[0]
    volumes = np.zeros((n_simp, ))
    c = np.zeros((vertices.shape[1], ))
    for i in range(n_simp):
        s_verts = vertices[simplicialation.simplices[i, :], :]
        vol = _3d_simplex_volume(s_verts)
        volumes[i] = vol
        c += vol * np.mean(s_verts, axis=0)

    total_volume = volumes.sum()
    centroid = c / total_volume
    return centroid, total_volume


def _3d_simplex_volume(vertices: np.ndarray) -> float:
    """
    Calculates the total volume of the tetrahedron generated by the vertices
    :param vertices: Must be a numpy array of shape (4, 3)
    :return: The volume of the tetrahedron generated by the vertices.
    """
    ab = vertices[1, :] - vertices[0, :]
    ac = vertices[2, :] - vertices[0, :]
    ad = vertices[3, :] - vertices[0, :]
    volume = np.abs(1.0 / 6.0 * np.linalg.det(np.vstack((ab, ac, ad))))
    return volume

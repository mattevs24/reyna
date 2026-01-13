import numpy as np
import matplotlib.pyplot as plt

from reyna.polymesher.three_dimensional._auxilliaries.abstraction import Domain3D, PolyMesh3D

# TODO: need to work on this code a little here -- also include the plotting functions here, same as 2D just extended.


def write_to_vtk(filepath: str, poly_mesh: PolyMesh3D, domain: Domain3D) -> None:
    """
    This function writes a (n, 3) numpy array to a VTK file particularly with a focus on use in Paraview.

    Args:
        filepath: Path to the output VTK file
        poly_mesh: PolyMesh3D object
        domain: Domain3D object

    Returns:
        None
    """

    if filepath == 'execute':
        filepath = 'python_export.vtk'

    with open(filepath, 'w') as file:
        file.write('# vtk DataFile Version 3.0\n')
        file.write('VTK from Python\n')
        file.write('ASCII\n')
        file.write('DATASET POLYDATA\n')

        vertices = poly_mesh.vertices

        dimensions = vertices.shape[1]

        if dimensions == 2:
            vertices = np.concatenate((vertices, np.zeros((vertices.shape[0], 1), dtype=np.float)), axis=1)
        elif dimensions != 3:
            raise ValueError('"vertices" must have 2 or 3 dimensional points.')

        n_points = vertices.shape[0]

        file.write(f"POINTS {n_points} float\n")

        for i in range(n_points):
            file.write(f"{vertices[i, 0]} {vertices[i, 1]} {vertices[i, 2]}\n")

        file.write('\n')

        if dimensions == 2:

            n_elements = len(poly_mesh.filtered_regions)
            size = sum([len(element) for element in poly_mesh.filtered_regions]) + n_elements
            file.write(f"POLYGONS {n_elements} {size}\n")

            for i in range(n_elements):
                element = poly_mesh.filtered_regions[i]
                line = f"{len(element)}"
                for point in element:
                    line += f" {point}"

                line += '\n'
                file.write(line)
        else:
            tol = 1e-1
            valid_nodes = set(np.argwhere((domain.distances(poly_mesh.vertices)[:, -1] < tol)).flatten().tolist())
            valid_facets = [i for i, ridge in enumerate(poly_mesh.ridge_vertices) if set(ridge).issubset(valid_nodes)]

            n_elements = len(valid_facets)
            size = sum([len(facet) for i, facet in enumerate(poly_mesh.ridge_vertices) if i in valid_facets]) + n_elements
            file.write(f"POLYGONS {n_elements} {size}\n")

            # n_elements = len(poly_mesh.ridge_vertices)
            # size = sum([len(facet) for facet in poly_mesh.ridge_vertices]) + n_elements
            # file.write(f"POLYGONS {n_elements} {size}\n")

            for i, element in enumerate(poly_mesh.ridge_vertices):
                if i not in valid_facets:
                    continue
                line = f"{len(element)}"
                for point in element:
                    line += f" {point}"

                line += '\n'
                file.write(line)

        file.close()


def read_from_voro_pp(filepath: str) -> PolyMesh3D:
    # TODO: this function needs to take in all the values from voro++ and generate the mesh here.
    ...


def display_mesh(poly_mesh: PolyMesh3D) -> None:
    # TODO: idealy for this function, the facets of the mesh are alreacy known -- this was achieved in two dimensions
    #  simply by the way in which voronoi tesellations are contructed, but not here -- more work is required.

    ax = plt.figure().add_subplot(projection='3d')

    for facet in poly_mesh.facets:
        indecies = np.concatenate((np.arange(len(facet)), np.array([0])))
        face = np.array(facet)[indecies]
        ax.plot(
            poly_mesh.vertices[face, 0],
            poly_mesh.vertices[face, 1],
            poly_mesh.vertices[face, 2],
            color='r',
        )

    plt.show()
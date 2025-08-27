import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from reyna.polymesher.two_dimensional._auxilliaries.abstraction import PolyMesh


def display_mesh(poly_mesh: PolyMesh, **kwargs) -> None:
    """
    A function to display the generated PolyMesh.

    Args:
        poly_mesh (PolyMesh): A PolyMesh object.
        **kwargs: Additional keyword arguments. There are several valid inputs here. 'figsize', 'color_map' and
        'save_path' are as expected. Any further keyword arguments are passed as parameters into matplotlib.patches
        Polygon function.

    See Also:
        [matplotlib.patches.Polygon](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Polygon.html)

    """
    figsize: tuple = (8, 8)
    if 'figsize' in kwargs:
        figsize = kwargs.pop('figsize', (8, 8))

    color_map = None
    if 'color_map' in kwargs:
        color_map = kwargs.pop('color_map', None)

    save_path = None
    if 'save_path' in kwargs:
        save_path = kwargs.pop('save_path', None)

    color_scheme = kwargs

    if not color_scheme:
        color_scheme = {'alpha': 0.2, 'linewidth': 0.5, 'edgecolor': 'black', 'facecolor': 'grey'}

    assert (dimension := poly_mesh.filtered_points.shape[1]) == 2, \
        f"The dimension of the points must be equal to 2 to use this function: the dimension of " \
        f"the points inputted is {dimension}"

    fig, ax = plt.subplots(figsize=figsize)
    plt.axis('off')

    for i, region in enumerate(poly_mesh.filtered_regions):
        if color_map is not None:
            color_scheme['facecolor'] = color_map(poly_mesh.filtered_points[i, :])

        ax.add_patch(Polygon(poly_mesh.vertices[region, :], **color_scheme))

    ax.set_xlim(poly_mesh.domain.bounding_box[0, :])
    ax.set_ylim(poly_mesh.domain.bounding_box[1, :])

    if save_path is not None:
        plt.savefig(save_path, dpi=800)
    else:
        plt.show()

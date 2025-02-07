import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from enum import Enum

from reyna.polymesher.two_dimensional._auxilliaries.abstraction import PolyMesh


class ColorScheme(Enum):
    """
    An enumeration with a few common color schemes.
    """
    default = {"alpha": 0.2, "lw": 1, "edgecolor": "black", "facecolor": "blue"}
    black = {"alpha": 0.2, "lw": 0.5, "edgecolor": "black", "facecolor": "grey"}
    blue = {}
    green = {}
    red = {}


def display_mesh(poly_mesh: PolyMesh, **kwargs) -> None:

    """
    A function to display the generated PolyMesh

    Args:
        poly_mesh (PolyMesh): A PolyMesh object.

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

    if "color_scheme" in kwargs:
        color_scheme = kwargs.get("color_scheme")
        assert type(color_scheme) == ColorScheme, "Input 'color_scheme' must be of type ColorScheme."
        color_scheme = color_scheme.value

    if not color_scheme:
        color_scheme = ColorScheme.black.value

    assert (dimension := poly_mesh.filtered_points.shape[1]) == 2, \
        f"The dimension of the points must be equal to 2 to use this function: the dimention of " \
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

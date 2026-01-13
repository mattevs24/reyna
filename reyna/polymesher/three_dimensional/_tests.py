from reyna.polymesher.three_dimensional.main import poly_mesher_3d
from reyna.polymesher.three_dimensional.domains import CuboidDomain

import numpy as np

from reyna.polymesher.three_dimensional.tools import display_mesh

dom = CuboidDomain(bounding_box=np.array([[0, 5], [0, 2], [0, 3]]))
polygonal_mesh = poly_mesher_3d(dom, max_iterations=15, n_points=1000)

display_mesh(polygonal_mesh)

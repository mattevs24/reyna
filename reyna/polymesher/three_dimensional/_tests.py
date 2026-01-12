from reyna.polymesher.three_dimensional.main import poly_mesher_3d
from reyna.polymesher.three_dimensional.domains import CuboidDomain

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

dom = CuboidDomain(bounding_box=np.array([[0, 5], [0, 2], [0, 3]]))
polygonal_mesh = poly_mesher_3d(dom, max_iterations=10, n_points=1000)


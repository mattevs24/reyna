import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, Point

from reyna.polymesher.two_dimensional.domains import CircleDomain, RectangleDomain, CircleCircleDomain
from reyna.polymesher.two_dimensional.main import poly_mesher
from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry

from main import DGFEM
from plotter import plot_DG


# Section: advection testing -- tested and happy with this in all cases
#
# advection = lambda x: np.ones(x.shape, dtype=float)
# forcing = lambda x: np.pi * (np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
#                              np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]))
# solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
#
# n_elements = [32, 64, 128, 256, 512, 1024, 2048, 4096]
#
# h_s_dict = {}
# dg_norms_dict = {}
# l2_norms_dict = {}
# h1_norms_dict = {}
#
# for p in [1, 2, 3]:
#
#     h_s = []
#     dg_norms = []
#     l2_norms = []
#     h1_norms = []
#
#     for n_r in n_elements:
#
#         dom = RectangleDomain(np.array([[0.5, 1.5], [0.5, 1.5]]))
#         poly_mesh = poly_mesher(dom, max_iterations=50, n_points=n_r)
#         geometry = DGFEMGeometry(poly_mesh)
#
#         dg = DGFEM(geometry, polynomial_degree=p)
#         dg.add_data(
#             advection=advection,
#             dirichlet_bcs=solution,
#             forcing=forcing
#         )
#         dg.dgfem(solve=True)
#
#         l2_error, dg_error, _ = dg.errors(exact_solution=solution,
#                                           div_advection=lambda x: np.zeros(x.shape[0]))
#         dg_norms.append(float(dg_error))
#         l2_norms.append(float(l2_error))
#
#         _h = -np.inf
#         for element in geometry.mesh.filtered_regions:
#             poly = Polygon(geometry.nodes[element, :])
#             box = poly.minimum_rotated_rectangle
#             _x, _y = box.exterior.coords.xy
#             edge_length = (Point(_x[0], _y[0]).distance(Point(_x[1], _y[1])),
#                            Point(_x[1], _y[1]).distance(Point(_x[2], _y[2])))
#             _h = max(_h, max(edge_length))
#
#         h_s.append(_h)
#     # plot_DG(dg.solution, geometry, dg.polydegree)
#
#     h_s_dict[p] = h_s
#     dg_norms_dict[p] = dg_norms
#     l2_norms_dict[p] = l2_norms
#
# x_ = np.linspace(0.03, 0.3, 100)
#
# fig, axes = plt.subplots(1, 2)
#
# for k, v in dg_norms_dict.items():
#     axes[0].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[0].plot(x_, 10 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# axes[0].plot(x_, 5 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# axes[0].plot(x_, 2.0 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
#
# axes[0].legend(title='dG norm')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
#
# for k, v in l2_norms_dict.items():
#     axes[1].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[1].plot(x_, 4.0 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# axes[1].plot(x_, 0.5 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
# axes[1].plot(x_, 0.2 * x_ ** 4.0, linestyle='--', label=r'$h^{4}$')
#
# axes[1].legend(title='L2 norm')
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
#
# plt.show()

# Section: diffusion testing

# diffusion = lambda x: np.repeat([np.identity(2, dtype=float)], x.shape[0], axis=0)
# forcing = lambda x: 2.0 * np.pi ** 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
# solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
#
#
# def grad_solution(x: np.ndarray):
#     u_x = np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
#     u_y = np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
#
#     return np.vstack((u_x, u_y)).T
#
#
# n_elements = [32, 64, 128, 256, 512, 1024, 2048, 4096]
#
# h_s_dict = {}
# dg_norms_dict = {}
# l2_norms_dict = {}
# h1_norms_dict = {}
#
# for p in [1, 2, 3]:
#
#     h_s = []
#     dg_norms = []
#     l2_norms = []
#     h1_norms = []
#
#     for n_r in n_elements:
#
#         dom = RectangleDomain(np.array([[0.5, 1.5], [0.5, 1.5]]))
#         poly_mesh = poly_mesher(dom, max_iterations=100, n_points=n_r, cleaned=True)
#         geometry = DGFEMGeometry(poly_mesh)
#
#         dg = DGFEM(geometry, polynomial_degree=p)
#         dg.add_data(
#             diffusion=diffusion,
#             dirichlet_bcs=solution,
#             forcing=forcing
#         )
#         dg.dgfem(solve=True)
#
#         # plot_DG(dg.solution, geometry, dg.polydegree)
#
#         l2_error, dg_error, h1_error = dg.errors(exact_solution=solution,
#                                                  div_advection=lambda x: np.zeros(x.shape[0]),
#                                                  grad_exact_solution=grad_solution)
#         l2_norms.append(l2_error)
#         dg_norms.append(dg_error)
#         h1_norms.append(h1_error)
#
#         h_s.append(geometry.h)
#     # plot_DG(dg.solution, geometry, dg.polydegree)
#
#     h_s_dict[p] = h_s
#     dg_norms_dict[p] = dg_norms
#     l2_norms_dict[p] = l2_norms
#     h1_norms_dict[p] = h1_norms
#
# x_ = np.linspace(0.03, 0.3, 100)
#
# fig, axes = plt.subplots(1, 3)
#
# for k, v in dg_norms_dict.items():
#     axes[0].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[0].plot(x_, 10 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# axes[0].plot(x_, 5 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# axes[0].plot(x_, 2.0 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
#
# axes[0].legend(title='dG norm')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
#
# for k, v in h1_norms_dict.items():
#     axes[1].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[1].plot(x_, 4.0 * x_ ** 1.0, linestyle='--', label=r'$h^{1}$')
# axes[1].plot(x_, 0.5 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# axes[1].plot(x_, 0.2 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
#
# axes[1].legend(title='H1 norm')
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
#
# for k, v in l2_norms_dict.items():
#     axes[2].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[2].plot(x_, 4.0 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# axes[2].plot(x_, 0.5 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
# axes[2].plot(x_, 0.2 * x_ ** 4.0, linestyle='--', label=r'$h^{4}$')
#
# axes[2].legend(title='L2 norm')
# axes[2].set_xscale('log')
# axes[2].set_yscale('log')
#
# plt.show()


# Section: diffusion-advection-reaction

# diffusion = lambda x: np.repeat([np.identity(2, dtype=float)], x.shape[0], axis=0)
# advection = lambda x: np.ones(x.shape, dtype=float)
# reaction = lambda x: np.pi ** 2 * np.ones(x.shape[0], dtype=float)
# forcing = lambda x: np.pi * (np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
#                              np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])) + \
#                     3.0 * np.pi ** 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
#
# solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
#
#
# def grad_solution(x: np.ndarray):
#     u_x = np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
#     u_y = np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
#
#     return np.vstack((u_x, u_y)).T
#
#
# n_elements = [32, 64, 128, 256, 512, 1024, 2048, 4096]
#
# h_s_dict = {}
# dg_norms_dict = {}
# l2_norms_dict = {}
# h1_norms_dict = {}
#
# for p in [1, 2, 3]:
#
#     h_s = []
#     dg_norms = []
#     l2_norms = []
#     h1_norms = []
#
#     for n_r in n_elements:
#
#         dom = CircleCircleDomain()
#         poly_mesh = poly_mesher(dom, max_iterations=50, n_points=n_r, cleaned=True)
#         geometry = DGFEMGeometry(poly_mesh)
#
#         dg = DGFEM(geometry, polynomial_degree=p)
#         dg.add_data(
#             diffusion=diffusion,
#             advection=advection,
#             reaction=reaction,
#             dirichlet_bcs=solution,
#             forcing=forcing
#         )
#         dg.dgfem(solve=True)
#
#         if n_r > 8000:
#             plot_DG(dg.solution, geometry, dg.polydegree)
#
#         l2_error, dg_error, h1_error = dg.errors(exact_solution=solution,
#                                                  div_advection=lambda x: np.zeros(x.shape[0]),
#                                                  grad_exact_solution=grad_solution)
#         l2_norms.append(l2_error)
#         dg_norms.append(dg_error)
#         h1_norms.append(h1_error)
#         h_s.append(geometry.h)
#
#     h_s_dict[p] = h_s
#     dg_norms_dict[p] = dg_norms
#     l2_norms_dict[p] = l2_norms
#     h1_norms_dict[p] = h1_norms
#
# x_ = np.linspace(0.03, 0.3, 100)
#
# fig, axes = plt.subplots(1, 3)
#
# for k, v in dg_norms_dict.items():
#     axes[0].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[0].plot(x_, 10 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# axes[0].plot(x_, 5 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# axes[0].plot(x_, 2.0 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
#
# axes[0].legend(title='dG norm')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
#
# for k, v in h1_norms_dict.items():
#     axes[1].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[1].plot(x_, 4.0 * x_ ** 1.0, linestyle='--', label=r'$h^{1}$')
# axes[1].plot(x_, 0.5 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# axes[1].plot(x_, 0.2 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
#
# axes[1].legend(title='H1 norm')
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
#
# for k, v in l2_norms_dict.items():
#     axes[2].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[2].plot(x_, 4.0 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# axes[2].plot(x_, 0.5 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
# axes[2].plot(x_, 0.2 * x_ ** 4.0, linestyle='--', label=r'$h^{4}$')
#
# axes[2].legend(title='L2 norm')
# axes[2].set_xscale('log')
# axes[2].set_yscale('log')
#
# plt.show()

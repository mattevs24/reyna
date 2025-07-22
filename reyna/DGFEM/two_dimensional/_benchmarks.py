import time

import numpy as np
import matplotlib.pyplot as plt

from reyna.polymesher.two_dimensional.domains import CircleDomain, RectangleDomain, CircleCircleDomain
from reyna.polymesher.two_dimensional.main import poly_mesher
from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry

from main import DGFEM
from plotter import plot_DG

# Section: advection testing -- tested and happy with this in all cases

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
#         l2_error, dg_error, _, _ = dg.errors(exact_solution=solution,
#                                              div_advection=lambda x: np.zeros(x.shape[0]))
#         dg_norms.append(float(dg_error))
#         l2_norms.append(float(l2_error))
#
#         h_s.append(geometry.h)
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
# axes[0].plot(x_, x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# axes[0].plot(x_, 0.5 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# axes[0].plot(x_, 0.2 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
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

diffusion = lambda x: np.repeat([np.identity(2, dtype=float)], x.shape[0], axis=0)
reaction = lambda x: np.pi ** 2 * np.ones(x.shape[0], dtype=float)
forcing = lambda x: 3.0 * np.pi ** 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def grad_solution(x: np.ndarray):
    u_x = np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    u_y = np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

    return np.vstack((u_x, u_y)).T


# n_elements = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
n_elements = [32, 64, 128, 256, 512, 1024, 2048, 4096]

h_s_dict = {}
dg_norms_dict = {}
l2_norms_dict = {}
h1_norms_dict = {}

int_l2_norms_dict = {}
int_h1_norms_dict = {}
int_edges_norms_dict = {}
b_diff_norms_dict = {}
b_adv_norms_dict = {}

for p in [1, 2, 3]:

    h_s = []
    dg_norms = []
    l2_norms = []
    h1_norms = []

    int_l2_norms = []
    int_h1_norms = []
    int_edges_norms = []
    b_diff_norms = []
    b_adv_norms = []

    for n_r in n_elements:

        # dom = CircleCircleDomain()
        dom = RectangleDomain(np.array([[0.0, 1.0], [0.0, 1.0]]))
        poly_mesh = poly_mesher(dom, max_iterations=10, n_points=n_r, cleaned=True)
        geometry = DGFEMGeometry(poly_mesh)

        dg = DGFEM(geometry, polynomial_degree=p)
        dg.add_data(
            diffusion=diffusion,
            reaction=reaction,
            dirichlet_bcs=solution,
            forcing=forcing
        )
        dg.dgfem(solve=True)

        # plot_DG(dg.solution, geometry, dg.polydegree)

        l2_error, dg_error, h1_error = dg.errors(exact_solution=solution,
                                                 div_advection=lambda x: np.zeros(x.shape[0]),
                                                 grad_exact_solution=grad_solution)
        l2_norms.append(l2_error)
        dg_norms.append(dg_error)
        h1_norms.append(h1_error)
        h_s.append(geometry.h)

    h_s_dict[p] = h_s
    dg_norms_dict[p] = dg_norms
    l2_norms_dict[p] = l2_norms
    h1_norms_dict[p] = h1_norms

x_ = np.linspace(0.03, 0.3, 100)

fig, axes = plt.subplots(1, 3)

for k, v in dg_norms_dict.items():
    axes[0].plot(h_s_dict[p], v, label=f'P{k}')

axes[0].plot(x_, 15 * x_ ** 1.0, linestyle='--', label=r'$h^{1}$')
axes[0].plot(x_, 10 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
axes[0].plot(x_, 5 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
axes[0].plot(x_, 2.0 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')

axes[0].legend(title='dG norm')
axes[0].set_xscale('log')
axes[0].set_yscale('log')

for k, v in h1_norms_dict.items():
    axes[1].plot(h_s_dict[p], v, label=f'P{k}')

axes[1].plot(x_, 4.0 * x_ ** 1.0, linestyle='--', label=r'$h^{1}$')
axes[1].plot(x_, 0.5 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
axes[1].plot(x_, 0.2 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')

axes[1].legend(title='H1 norm')
axes[1].set_xscale('log')
axes[1].set_yscale('log')

for k, v in l2_norms_dict.items():
    axes[2].plot(h_s_dict[p], v, label=f'P{k}')

axes[2].plot(x_, 4.0 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
axes[2].plot(x_, 0.5 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
axes[2].plot(x_, 0.2 * x_ ** 4.0, linestyle='--', label=r'$h^{4}$')

axes[2].legend(title='L2 norm')
axes[2].set_xscale('log')
axes[2].set_yscale('log')

plt.show()

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
#
# h_s_dict = {}
# dg_norms_dict = {}
# l2_norms_dict = {}
# h1_norms_dict = {}
#
# int_norms_dict = {}
# int_edges_norms_dict = {}
# b_diff_norms_dict = {}
# b_adv_norms_dict = {}
#
#
# for p in [1, 2, 3]:
#
#     h_s = []
#     dg_norms = []
#     l2_norms = []
#     h1_norms = []
#
#     int_norms = []
#     int_edges_norms = []
#     b_diff_norms = []
#     b_adv_norms = []
#
#     for n_r in n_elements:
#
#         dom = CircleDomain(np.array([[-1, 1], [-1, 1]]))
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
#         l2_error, dg_error, h1_error, subnorm_dict = dg.errors(exact_solution=solution,
#                                                                div_advection=lambda x: np.zeros(x.shape[0]),
#                                                                grad_exact_solution=grad_solution)
#
#         l2_norms.append(l2_error)
#         dg_norms.append(dg_error)
#         h1_norms.append(h1_error)
#         h_s.append(geometry.h)
#
#         int_norms.append(subnorm_dict['int'])
#         int_edges_norms.append(subnorm_dict['int_edges'])
#         b_diff_norms.append(subnorm_dict['b_diff'])
#         b_adv_norms.append(subnorm_dict['b_adv'])
#
#     h_s_dict[p] = h_s
#     dg_norms_dict[p] = dg_norms
#     l2_norms_dict[p] = l2_norms
#     h1_norms_dict[p] = h1_norms
#
#     int_norms_dict[p] = int_norms
#     int_edges_norms_dict[p] = int_edges_norms
#     b_diff_norms_dict[p] = b_diff_norms
#     b_adv_norms_dict[p] = b_adv_norms
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
# axes[0].plot(x_, 10 * x_ ** 1.0, linestyle='--', label=r'$h^{1}$')
# axes[0].plot(x_, 5 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# axes[0].plot(x_, 2.0 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
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
#
# fig, axes = plt.subplots(1, 4)
#
# for k, v in int_edges_norms_dict.items():
#     axes[0].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[0].plot(x_, 10 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# axes[0].plot(x_, 5 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# axes[0].plot(x_, 2.0 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
#
# axes[0].legend(title='int_edges')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
#
# for k, v in b_adv_norms_dict.items():
#     axes[1].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[1].plot(x_, 0.1 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# axes[1].plot(x_, 0.05 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# axes[1].plot(x_, 0.02 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
#
# axes[1].legend(title='b_adv')
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
#
# for k, v in b_diff_norms_dict.items():
#     axes[2].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[2].plot(x_, 10 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# axes[2].plot(x_, 5 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# axes[2].plot(x_, 2.0 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
#
# axes[2].legend(title='B_diff')
# axes[2].set_xscale('log')
# axes[2].set_yscale('log')
#
# for k, v in int_norms_dict.items():
#     axes[3].plot(h_s_dict[p], v, label=f'P{k}')
#
# axes[3].plot(x_, 30 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# axes[3].plot(x_, 10 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
# axes[3].plot(x_, 2 * x_ ** 4.0, linestyle='--', label=r'$h^{4}$')
#
# axes[3].legend(title='L2 norm')
# axes[3].set_xscale('log')
# axes[3].set_yscale('log')
#
# plt.show()

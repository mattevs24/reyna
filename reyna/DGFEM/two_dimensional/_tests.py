import time

import numpy as np

from reyna.polymesher.two_dimensional.domains import RectangleDomain, HornDomain, CircleCircleDomain, LShapeDomain
from reyna.polymesher.two_dimensional.main import poly_mesher

from main import DGFEM
from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry
from plotter import plot_DG

np.set_printoptions(linewidth=400)

solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

advection = lambda x: np.ones(x.shape, dtype=float)
forcing = lambda x: np.pi * (np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
                             np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]))


def grad_u_exact(x: np.ndarray):
    u_x = np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    u_y = np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
    return np.vstack((u_x, u_y)).T

# Section: advection testing - Square Domain

import cProfile

dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
poly_mesh = poly_mesher(dom, max_iterations=5, n_points=1024)
geometry = DGFEMGeometry(poly_mesh)

dg = DGFEM(geometry, polynomial_degree=2)
dg.add_data(advection=advection, dirichlet_bcs=solution, forcing=forcing)
# dg.dgfem(solve=True)

cProfile.run('dg.dgfem(solve=True)', sort='cumtime')

# plot_DG(dg.solution, geometry, dg.polydegree)

raise KeyboardInterrupt

# Section: advection testing - Square domain, shifted
#
# dom = RectangleDomain(np.array([[0, 1], [0, 1]]))
# poly_mesh = poly_mesher(dom, max_iterations=5, n_points=500)
# poly_mesh = poly_mesher_cleaner(poly_mesh)
# geometry = DGFEMGeometry(poly_mesh)
#
# dg = DGFEM(geometry, polynomial_degree=1)
# dg.add_data(advection=advection, dirichlet_bcs=solution, forcing=forcing)
# dg.dgfem(solve=True)
#
# plot_DG(dg.solution, geometry, dg.polydegree)


# Section: advection testing - CircleCircle

# dom = CircleCircleDomain()
# poly_mesh = poly_mesher(dom, max_iterations=5, n_points=1000)
# poly_mesh = poly_mesher_cleaner(poly_mesh)
# geometry = DGFEMGeometry(poly_mesh)
#
# dg = DGFEM(geometry, polynomial_degree=1)
# dg.add_data(advection=advection, dirichlet_bcs=solution, forcing=forcing)
# dg.dgfem(solve=True)
#
# plot_DG(dg.solution, geometry, dg.polydegree)


# Section: diuffusion testing - Square
#
# diffusion = lambda x: np.repeat([np.identity(2, dtype=float)], x.shape[0], axis=0)
# forcing = lambda x: 2 * np.pi ** 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
#
# dom = RectangleDomain(np.array([[-1, 1], [-1, 1]]))
# poly_mesh = poly_mesher(dom, max_iterations=5, n_points=1000)
# poly_mesh = poly_mesher_cleaner(poly_mesh)
# geometry = DGFEMGeometry(poly_mesh)
#
#
# dg = DGFEM(geometry, polynomial_degree=1)
# dg.add_data(diffusion=diffusion, dirichlet_bcs=solution, forcing=forcing)
# dg.dgfem(solve=True)
#
# plot_DG(dg.solution, geometry, dg.polydegree)
#
# dg_norm, l2_norm, _ = dg.errors(
#     exact_solution=solution,
#     grad_exact_solution=grad_u_exact,
#     div_advection=lambda x: np.zeros(x.shape[0])
# )
#
# print(f"dg norm: {dg_norm}")
# print(f"L2 norm: {l2_norm}")

# Section: diuffusion testing - CircleCircle
#
# diffusion = lambda x: np.repeat([np.identity(2, dtype=float)], x.shape[0], axis=0)
# forcing = lambda x: 2 * np.pi ** 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
#
# dom = CircleCircleDomain()
# poly_mesh = poly_mesher(dom, max_iterations=5, n_points=1000)
# poly_mesh = poly_mesher_cleaner(poly_mesh)
# geometry = DGFEMGeometry(poly_mesh)
#
#
# dg = DGFEM(geometry, polynomial_degree=1)
# dg.add_data(diffusion=diffusion, dirichlet_bcs=solution, forcing=forcing)
# dg.dgfem(solve=True)
#
# plot_DG(dg.solution, geometry, dg.polydegree)
#
# dg_norm, l2_norm, _ = dg.errors(
#     exact_solution=solution,
#     grad_exact_solution=grad_u_exact,
#     div_advection=lambda x: np.zeros(x.shape[0])
# )
#
# print(f"dg norm: {dg_norm}")
# print(f"L2 norm: {l2_norm}")

# Section: diuffusion-advection-reaction testing

dom = HornDomain()
poly_mesh = poly_mesher(dom, max_iterations=10, n_points=1024)
geometry = DGFEMGeometry(poly_mesh)

diffusion = lambda x: np.repeat([np.identity(2, dtype=float)], x.shape[0], axis=0)
advection = lambda x: np.ones(x.shape, dtype=float)
reaction = lambda x: np.pi ** 2 * np.ones(x.shape[0], dtype=float)
forcing = lambda x: (np.pi * (np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
                             np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])) +
                     3.0 * np.pi ** 2 * np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]))

polydegree = 3

dg = DGFEM(geometry, polynomial_degree=polydegree)
dg.add_data(diffusion=diffusion, advection=advection, reaction=reaction, dirichlet_bcs=solution, forcing=forcing)

_old_time = 4.770437955856323  if polydegree == 1 else 19.09006381034851  # These are P1 and P3 cases
_time = time.time()

dg.dgfem(solve=True)

# TODO: need to make all the changes and benchmark properly before commiting to main and updating the package

print(f"Time saved (s): {(_old_time - (time.time() - _time)):.5f}s")
print(f"Time saved (%): {(100 * (_old_time - (time.time() - _time)) / _old_time):.5f}%")

plot_DG(dg.solution, geometry, dg.polydegree)
#
# dg_norm, l2_norm, _ = dg.errors(
#     exact_solution=solution,
#     grad_exact_solution=grad_u_exact,
#     div_advection=lambda x: np.zeros(x.shape[0])
# )
#
# print(f"dg norm: {dg_norm}")
# print(f"L2 norm: {l2_norm}")

# Section: log of time changes

#     Original           stiffness          Int_edges           Diff_assembly       Adv_assembly
# P1  4.770437955856323  4.246790170669556  3.5955491065979004  3.5803170204162598  3.498729944229126
# P3  19.09006381034851  15.20890784263611  10.651919126510620  10.508561134338379  10.51161813735962

#     F_assembly         Adv_diff_bcs
# P1  3.480814933776855  3.420440912246704
# P3  9.955646991729736  9.842134952545166

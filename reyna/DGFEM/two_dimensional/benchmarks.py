import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon, Point

from reyna.polymesher.two_dimensional.domains import CircleCircleDomain
from reyna.polymesher.two_dimensional.main import poly_mesher, poly_mesher_cleaner
from reyna.geometry.two_dimensional.DGFEM import DGFEMGeometry

from main import DGFEM
# from plotter import plot_DG


# Section: advection testing

# advection = lambda x: np.ones(x.shape, dtype=float)
# forcing = lambda x: np.pi * (np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
#                              np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]))
# solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

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

# Section: diffusion-advection-reaction

# TODO: dg norm not converging fast enough? may be the elliptic and hyperbolic boundary

diffusion = lambda x: np.repeat([np.identity(2, dtype=float)], x.shape[0], axis=0)
advection = lambda x: np.ones(x.shape, dtype=float)
reaction = lambda x: -2 * np.pi ** 2 * np.ones(x.shape[0], dtype=float)
forcing = lambda x: np.pi * (np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1]) +
                             np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]))
solution = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])


def grad_solution(x: np.ndarray):
    u_x = np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
    u_y = np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

    return np.vstack((u_x, u_y)).T


n_elements = [32, 64, 128, 256, 512, 1024, 2048]

h_s_dict = {}
dg_norms_dict = {}
l2_norms_dict = {}
h1_norms_dict = {}

for p in [1, 2, 3]:

    h_s = []
    dg_norms = []
    l2_norms = []
    h1_norms = []

    for n_r in n_elements:

        dom = CircleCircleDomain()
        poly_mesh = poly_mesher(dom, max_iterations=10, n_points=n_r)
        poly_mesh = poly_mesher_cleaner(poly_mesh)
        geometry = DGFEMGeometry(poly_mesh)

        dg = DGFEM(geometry, polynomial_degree=p)
        dg.add_data(
            diffusion=diffusion,
            dirichlet_bcs=solution,
            forcing=forcing
        )
        dg.dgfem(solve=True)

        # plot_DG(dg.solution, geometry, dg.polydegree)

        dg_error, l2_error, h1_error = dg.errors(exact_solution=solution,
                                                 div_advection=lambda x: np.zeros(x.shape[0]),
                                                 grad_exact_solution=grad_solution)
        dg_norms.append(float(dg_error))
        l2_norms.append(float(l2_error))
        h1_norms.append(h1_error)

        _h = -np.inf
        for element in geometry.mesh.filtered_regions:
            poly = Polygon(geometry.nodes[element, :])
            box = poly.minimum_rotated_rectangle
            _x, _y = box.exterior.coords.xy
            edge_length = (Point(_x[0], _y[0]).distance(Point(_x[1], _y[1])),
                           Point(_x[1], _y[1]).distance(Point(_x[2], _y[2])))
            _h = max(_h, max(edge_length))

        h_s.append(_h)
    # plot_DG(dg.solution, geometry, dg.polydegree)

    h_s_dict[p] = h_s
    dg_norms_dict[p] = dg_norms
    l2_norms_dict[p] = l2_norms
    h1_norms_dict[p] = h1_norms

x_ = np.linspace(0.03, 0.3, 100)

fig, axes = plt.subplots(1, 3)

for k, v in dg_norms_dict.items():
    axes[0].plot(h_s_dict[p], v, label=f'P{k}')

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


# # Section: hyperbolic test case results.
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.667471351503544, 0.5797567060215092, 0.5101411608005707, 0.44115079959517794, 0.3701710038011913,
#           0.31608574527296457, 0.26729828955744234, 0.22675874493506334], label='DG0')
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.16109281486902058, 0.0930907808225883, 0.055816413546942806, 0.03350573399031336, 0.01910941187710968,
#           0.011779315968001148, 0.006794677273741668, 0.003893670586911219], label='DG1')
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.020893829363908916, 0.008634576100181203, 0.0037229604606629483, 0.0015751347244834708,
#           0.0006552608374984257, 0.00026572550792115297, 0.00011217282884066979, 4.837940576883884e-05], label='DG2')
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.002203747395022301, 0.0006854861000076535, 0.00017483665049941263, 6.128350352233879e-05,
#           4.405704932013318e-05, 5.1759744589116435e-05, 6.335568143827539e-05, 7.596458250613108e-05], label='DG3')
#
#
# x_ = np.linspace(0.03, 0.47, 100)
# plt.plot(x_, x_ ** 0.5, linestyle='--', label=r'$h^{\frac{1}{2}}$')
# plt.plot(x_, 0.5 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# plt.plot(x_, 0.15 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# plt.plot(x_, 0.05 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
#
#
# plt.legend()
# plt.yscale('log')
# plt.xscale('log')
# plt.show()
#
# # Section: diffusion benchmarking
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [1.702094150615805, 1.204151515211308, 0.8127042922878338, 0.5463623509352719, 0.31625560186873247,
#           0.21998462617388914, 0.14157858517975347, 0.09018851141945235], label='DG1')
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.2637874859608693, 0.12615965280019004, 0.0613149731567567, 0.029811761360479676, 0.012833299801693888,
#           0.006094685164645065, 0.0030478097159380543, 0.0015919965996815976], label='DG2')
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.031110300156962797, 0.010677417691704187, 0.0032191458297148355, 0.001128729865520778,
#           0.00033109120215039835, 0.00013435457924242008, 5.3840774648447445e-05, 0.0009779401045052908], label='DG3')
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.2971260113183066, 0.1828029058724379, 0.07861080302328909, 0.04195861622478331, 0.013787611777692114,
#           0.005984351422380635, 0.0023397120736058526, 0.0009185292937760148], label='DG1 (L2)')
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.010629481152984902, 0.003659443908576544, 0.000997738365926374, 0.0003009125098002091,
#           9.612316685790496e-05, 2.9803867011379028e-05, 9.667794789759899e-06,
#           4.142688284155784e-06], label='DG2 (L2)')
#
# plt.plot([0.3112956805706241, 0.27203498241298707, 0.17659926376169954, 0.1382651100773942, 0.09806406440271342,
#           0.07166092004527838, 0.04999316531748002, 0.0356082213421404],
#          [0.0007607702040317033, 0.00027817538451402086, 4.6606303854261645e-05, 1.3103531073610826e-05,
#           5.962159252399361e-06, 5.533886868971353e-06, 5.50276680979746e-06, 5.606683533554114e-06], label='DG3 (L2)')
#
#
# x_ = np.linspace(0.03, 0.32, 100)
# plt.plot(x_, 5 * x_ ** 1.0, linestyle='--', label=r'$h$')
# plt.plot(x_, x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# plt.plot(x_, 0.2 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
# plt.plot(x_, 0.02 * x_ ** 4.0, linestyle='--', label=r'$h^{4}$')
#
#
# plt.legend()
# plt.yscale('log')
# plt.xscale('log')
# plt.show()

# Section: A-R-D results

# x_ = np.linspace(0.03, 0.3, 100)
#
# fig, axes = plt.subplots(1, 3)
#
# axes[0].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [2.346634332908426, 1.7937323299955852, 1.1387680917394727, 0.7105605063374989, 0.4685159458674491, 0.30113002756704943, 0.20370196943122823], label='P1')
# axes[0].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [0.4183563146088462, 0.21501541161234874, 0.10450849358764143, 0.04524211627546161, 0.022683022052941256, 0.010474201707046472, 0.04280536334859797], label='P2')
# axes[0].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [0.061161789241129534, 0.03126735062706602, 0.007318307192585631, 0.002124769365151173, 0.0006880296895483801, 0.00023020433184232867, 0.0007551007451436769], label='P3')
#
# axes[0].plot(x_, 10 * x_ ** 1.5, linestyle='--', label=r'$h^{\frac{3}{2}}$')
# axes[0].plot(x_, 5 * x_ ** 2.5, linestyle='--', label=r'$h^{\frac{5}{2}}$')
# axes[0].plot(x_, 2.0 * x_ ** 3.5, linestyle='--', label=r'$h^{\frac{7}{2}}$')
#
# axes[0].legend(title='dG norm')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
#
# axes[1].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [1.7731409626445034, 1.3812591405586117, 1.0187429282967113, 0.5386068729905321, 0.36218541734977483, 0.24233767009847654, 0.1694210809252341], label='P1')
# axes[1].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [0.3346935689973194, 0.1625845510976792, 0.07505682298785475, 0.033517352043957074, 0.01615930924684276, 0.007565555468017206, 0.027305761778721013], label='P2')
# axes[1].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [0.051231084932843514, 0.026399131295240616, 0.006398374711499523, 0.0018036939192357135, 0.0005658948697874793, 0.00019107762441267982, 0.00043824534722527504], label='P3')
#
# axes[1].plot(x_, 4.0 * x_ ** 1.0, linestyle='--', label=r'$h^{1}$')
# axes[1].plot(x_, 0.5 * x_ ** 2.0, linestyle='--', label=r'$h^{2}$')
# axes[1].plot(x_, 0.2 * x_ ** 3.0, linestyle='--', label=r'$h^{3}$')
#
# axes[1].legend(title='H1 norm')
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
#
# axes[2].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [0.23623885045468193, 0.19096058424504028, 0.1370689633801626, 0.04687831192010336, 0.022566382595534683, 0.00950315734005893, 0.004447376541899582], label='P1')
# axes[2].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [0.029774049949736342, 0.010321870921610297, 0.003406346396491229, 0.0006982068288817749, 0.0001841233533952441, 4.923112889862657e-05, 0.00010222364385218264], label='P2')
# axes[2].plot(
#     [0.5240679570470221, 0.3439711004492267, 0.318375974752493, 0.1734388172008865, 0.11940086683160693, 0.08219779340885558, 0.059914707705453964],
#     [0.0024082003698252755, 0.0011381702121227484, 0.00017807356165610148, 2.6259385099624214e-05, 4.841136440383795e-06, 1.0747657691725399e-06, 1.3464769710841287e-06], label='P3')
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

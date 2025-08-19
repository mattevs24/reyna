import time

import numpy as np
import matplotlib.pyplot as plt

from reyna.polymesher.two_dimensional.domains import RectangleDomain, HornDomain, CircleCircleDomain, LShapeDomain
from reyna.polymesher.two_dimensional.main import poly_mesher

from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import (reference_to_physical_t3,
                                                                    tensor_tensor_leg, tensor_gradtensor_leg)
from reyna.DGFEM.two_dimensional._auxilliaries.assembly_aux import quad_GJ1 as orig_quad_GJ1


from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi


def quad_GL(n):
    """Gauss-Legendre quadrature on [-1,1]"""
    x, w = leggauss(n)
    return w, x


def quad_GJ1(n):
    """Gauss-Jacobi quadrature (alpha=0, beta=1) on [-1,1]"""
    x, w = roots_jacobi(n, 1, 0)
    return w, x


polydegree = 1

quadrature_order = 2 * polydegree + 1
w_x, x = quad_GJ1(quadrature_order)
w_y, y = quad_GL(quadrature_order)

quad_x = np.reshape(np.repeat(x, w_y.shape[0]), (-1, 1))
quad_y = np.reshape(np.tile(y, w_x.shape[0]), (-1, 1), order='F')
weights = (w_x[:, None] * w_y).flatten().reshape(-1, 1)

# The duffy points and the reference triangle points.
shiftpoints = np.hstack((0.5 * (1.0 + quad_x) * (1.0 - quad_y) - 1.0, quad_y))
ref_points = 0.5 * shiftpoints + 0.5

# plt.scatter(ref_points[:, 0], ref_points[:, 1])
# plt.show()

points = lambda _h: np.array([[0.0, 0.0], [_h, 0.0], [0.0, _h]])

error_x0, error_x1, error_x2, error_x3, error_x4 = [], [], [], [], []
true_solution = lambda _p, _h: _h ** (_p + 2) / ((_p + 1) * (_p + 2))


nodes = points(1.0)
P_Qpoints = reference_to_physical_t3(nodes, ref_points)

B = 0.5 * np.vstack((nodes[1, :] - nodes[0, :], nodes[2, :] - nodes[0, :]))
De_tri = np.abs(np.linalg.det(B))

p_errors = []

for p in np.linspace(0, 10, 11):
    _x = P_Qpoints[:, 0] ** p

    p_errors.append(np.abs(0.5 * np.dot(De_tri * weights.flatten(), _x) - true_solution(p, 1.0)))


plt.plot(np.linspace(0, 10, 11), p_errors, label='P')
plt.yscale('log')
plt.legend()
plt.show()

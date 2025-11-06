<!-- Add a small introduction here to being with -->

<!-- show the adr equation again here -->

This subpackage contains the discontinuous Galerkin finite element solvers. Currently,
we have the two dimensional code in place and running as intended. Please click to
the module to see all the details. There are a few things to note, these are discussed
in the remainder of this description.

## What does the solver do theoretically?

<!-- show all the weak form stuff again -->

This solver follows the methodology of Cangiani et al. [^1] closely. I.e. solves 
the problem of the form;

$$
\begin{split}
-\nabla\cdot(a\nabla u) + b\cdot\nabla u + cu &= f \quad \text{  on }\Omega\\
u &= g_D \quad\text{on } \Gamma := \partial\Omega
\end{split}
$$

The variational/weak form of this equation is derived and a discrete form 
is found. This converts to the problem; Find $u_h\in\mathbb{V}_p$ such that

$$
B(u_h, v_h) = \ell(v_h) \quad \forall v_h\in\mathbb{V}_p
$$

where,

$$
\begin{split}
B(u_h, v_h) &:= \sum_{\kappa\in\mathcal{T}_h}\int_{\kappa} \left(a\nabla u_h\cdot\nabla v_h + (b\cdot\nabla u_h) v_h + c u_hv_h\right)d\vec{x} \\
            &\quad-\int_{\Gamma^{\text{int}}\cup\Gamma}\left(\{a\nabla u_h\}\cdot[v_h] + \{a\nabla v_h\}\cdot[u_h] - \sigma_{\text{IP}}[u_h]\cdot[v_h]\right)ds \\
            &\quad- \sum_{\kappa\in\mathcal{T}_h}\left(\int_{\partial_-\kappa\setminus\Gamma}(b\cdot\vec{\nu})\lfloor u_h\rfloor v_h ds + \int_{\partial_-\kappa\cap\Gamma_-}(b\cdot\vec{\nu})u_hv_hds \right)
\end{split}
$$

and

$$
\ell(v_h) := \int_{\Omega}fv_hd\vec{x} - \int_{\Gamma_-}(b\cdot\vec{\nu})g_{\text{D}}v_hds - \int_{\Gamma}g_{\text{D}}\left(a\nabla v_h \cdot \vec{\nu} - \sigma_{\text{IP}} v_h\right) ds
$$

## What practical things do I need to know?

<!-- discuss practicalities! like numba functionalilty etc.... -->

There are several practical notes to make here. The code is written 
to be readable to all users. This allows learning and adaptability to be a focus.
Additionally, the code is fully vectorised. This allows speed and efficiency. 

There are some limitations and benefits to this 

- The solver is only able to support Dirichlet boundary conditions,
- It can accept any geometry and polynomial degree ($>=1$), 
- The coefficients must be ```numba``` compatible. ```numba``` allows the coefficients to be calculated rapidly, especially as they
are vector-valued themselves. Additionally, complicated source terms are commonplace
in DGFEM; this negates some of that cost.

There are several more benefits and limitations, but these are function specific and are explained in the 
relevent docstrings/documentation.

<!-- add citations to the dg book etc -->

[^1]: Cangiani, Andrea and Dong, Zhaonan and Georgoulis, Emmanuil H. and Houston, Paul. hp-Version Discontinuous Galerkin Methods on Polygonal and Polyhedral Meshes

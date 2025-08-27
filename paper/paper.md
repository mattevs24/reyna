---
title: 'Reyna: A minimal overhead, vectorised, polygonal discontinuous Galerkin finite element library.'
tags:
  - Python
  - Discontinuous Galerkin
  - Finite Elements
  - Polygonal
authors:
  - name: Matthew Evans
  - affiliation: 1
affiliations:
 - name: Department of Mathematical Sciences, University of Bath, United Kingdom
   index: 1
date: 24 August 2025
bibliography: paper.bib
---

# Summary

Modelling partial differential equations (PDEs) is a critical part of the modern world, whether the applications lie in 
engineering or natural sciences for example. Numerical schemes are often written in compiled languages for optimisation 
but these lack in practicality for experiementation or their prohibitive learning curve. Reyna is a Python package aimed
at targetting this and aiding in the experimentation and quick implementation of polygonal discontinuous Galerkin 
schemes. It has a flexible architechture, similar to other PDE solvers, making it easy to adapt and modify for a wide 
range of applications.

reyna is a Python library for solving partial differential equations (PDEs) using the polygonal discontinuous Galerkin
(PolyDG) method. Unlike traditional finite element approaches that restrict meshes to triangles or quadrilaterals, reyna 
allows computations directly on general polygonal meshes. This flexibility makes it easier to discretize complex 
geometries, perform adaptive mesh refinement, and efficiently capture features such as shocks or cracks that occur in 
many real-world problems. By combining minimal overhead with vectorized NumPy operations, reyna balances computational
efficiency with the simplicity and readability of pure Python, making advanced numerical methods more accessible for 
teaching, prototyping, and research.

# Statement of Need

Reyna was designed and developed to solve a wide range of PDEs with an emphasis on PDEs that appear often in nature [add 
citations here], neutronics [add citations here] and engineering [add citations here]. The simplistic, yet flexible, 
framework allows further developement and incorporation of more complicated PDEs and PDE systems with little work, 
enabling rapid research progress and experiementation.

A small selection of discontinuous Galerkin solvers admit polygonal discretiations of the computational domain but none
provide the simplicity of reyna through their use of Python


Discontinuous Galerkin (DG) methods are prized for their flexibility in handling complex geometries, high-order accuracy, 
and compatibility with discontinuities. Polygonal DG (PolyDG) extends this by allowing general polygonal and polyhedral 
elements, offering greater meshing flexibility and efficiency, especially for irregular domains.

Despite these advantages, no lightweight, vectorized, Python-native PolyDG library exists. Most implementations are 
embedded in large C++ or Fortran codes, lacking accessibility for prototyping, teaching, or rapid development. reyna 
fills this gap—providing efficient, easy-to-use, polygonal DG functionality in Python, with minimal boilerplate and 
maximal clarity.


Many scientific and engineering problems—ranging from fluid dynamics to material science—are governed by PDEs with 
solutions that exhibit discontinuities or evolve in highly irregular domains. The discontinuous Galerkin method is 
particularly well-suited to such problems because it allows for discontinuities between elements and preserves local 
conservation laws. Extending DG to polygonal meshes adds further versatility, enabling users to avoid costly mesh 
refinements and work with domains that cannot be easily partitioned into simple shapes.

Existing DG frameworks are often written in C++ or Fortran and integrated into large software ecosystems, which can be 
difficult for newcomers to install, extend, or adapt for experimental ideas. reyna addresses this gap by providing a 
lightweight, Python-native PolyDG implementation that can be used directly within the scientific Python stack. It is 
particularly valuable for researchers who need a rapid prototyping environment, and for educators who want to expose 
students to state-of-the-art discretization techniques without the complexity of a large HPC codebase.

# Description

reyna is designed around a small but powerful set of abstractions: meshes, basis functions, assembly routines, and 
solvers. Meshes can be built from arbitrary polygons, either generated procedurally (e.g., Voronoi diagrams) or imported 
from external meshing tools. Local basis functions and numerical quadrature rules are automatically adapted to each 
polygonal element. The assembly process is fully vectorized, ensuring that matrix and vector construction scales 
efficiently even for large meshes.

The solver interface integrates seamlessly with SciPy’s sparse linear algebra capabilities, allowing users to solve 
steady-state or time-dependent PDEs with minimal additional coding. Visualization utilities built on Matplotlib enable 
quick inspection of meshes and solutions, lowering the barrier to entry for experimentation. The overall design 
philosophy is clarity first: a small, focused codebase with minimal dependencies (NumPy, SciPy, Shapely, and 
Matplotlib), written so that users can readily inspect and modify the implementation.


reyna is organized around a set of modular building blocks that mirror the structure of a discontinuous Galerkin (DG) 
solver. At its core is a mesh representation that supports arbitrary polygonal elements. This generality makes it 
straightforward to import meshes from external meshing tools, generate unstructured tessellations such as Voronoi 
diagrams, or refine selected regions of a domain without disrupting the rest of the grid.

On top of the mesh, reyna provides routines for constructing local polynomial basis functions and numerical quadrature 
rules, which adapt automatically to the geometry of each polygonal element. These local operators are combined into 
global stiffness and mass matrices through highly vectorized assembly procedures. By relying on NumPy array operations 
rather than Python loops, the package achieves good performance while remaining compact and easy to inspect.
For solving PDEs, reyna exposes a clean interface that integrates with SciPy’s sparse linear algebra back end. Users can 
assemble and solve stationary problems (e.g., Poisson’s equation) or extend the same framework to time-dependent PDEs 
using their preferred ODE integrators. The package also includes utilities for defining boundary conditions, applying 
source terms, and post-processing results.

To aid exploration and teaching, reyna bundles simple visualization tools based on Matplotlib. Users can quickly plot 
meshes, inspect basis functions, or render computed solutions directly in a notebook or script. The combination of 
polygonal flexibility, DG accuracy, and Python accessibility makes reyna a useful platform both for researchers 
experimenting with new discretization strategies and for educators demonstrating advanced numerical methods.

# Example: 50 lines of DGFEM

# Acknowledgements

ME would like to acknowledge the support of Ansar Calloo, François Madiot, Tristan Pryer and Luca Zanetti throughout
this work for thier insights into both the mathematics behind discontinuous Galerkin methods as well as scientific
and research programming.

ME is supported by scholarships from the Statistical Applied Mathematics at Bath (SAMBa) EPSRC Centre for Doctoral 
Training (CDT) at the University of Bath under the project EP/S022945/1. ME is also partially funded by the French 
Alternative Energies and Atomic Energy Commission (CEA). All of this support is gratefully acknowledged.

# References

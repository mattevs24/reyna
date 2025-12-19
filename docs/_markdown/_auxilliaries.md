In this subpackage, we have two modules: ```abstraction``` and ```distance_functions```. Each of these
has a very specific purpose when generating and defining new domains. In ```abstraction```, we see two functions;
```PolyMesh``` and ```Domain```. A ```Domain``` object must be placed into ```poly_mesher``` to function as expected and
a ```PolyMesh``` object is outputted.

When generating custom meshes, one may bypass the use of ```poly_mesher``` entirely. If one requires the use of 
```DGFEMGeometry``` still however, the resulting mesh must be compatible with the ```PolyMesh``` object.

If one wants to create just a custom domain, one may use some of the already built-in distance functions to speed up
the design process. As in the original PolyMesher code by Talishi et al. [^1], these functions are used to describe the
distance a given point is from each of the boundary facets. One may look at the built-in domains to see how these functions
interact with each other in the final setting.

Custom meshes also become easier to define in this context; one may define a domain with fixed points, which remain the same
under the ```poly_mesher``` algorithm (see ```examples/discontinuous_solutions.ipynb``` for more information on applying
this concept).

[^1]: Talischi, C., Paulino, G.H., Pereira, A. et al. (2012). PolyMesher: a general-purpose mesh generator for polygonal elements written in Matlab. Struct Multidisc Optim 45, 309–328.
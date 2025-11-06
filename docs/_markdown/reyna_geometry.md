The ```geometry``` subpackage of this package is aimed at generating all the additional information about
the mesh which the solver needs to be able to run. Variables, such as which edges are interior and 
boundary edges and their corresponding normals are not part of the default ```PolyMesh``` implementation.
This was specifically done to allow for independent mesh design without the added costs of generating the
corresponding additional mesh variables.

As with the rest of the package, we present the ```two_dimensional``` geometry generation for the ```DGFEM```
solver. Additional geometries may be added, corresponding to various additonal schemes, but this is future work.
The current ```DGFEMGeometry``` object is optimised to generate the geometry required for the current
```DGFEM``` solver. More information on what the object generates can be found in the specific module.

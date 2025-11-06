Polymesher was originally a MatLab package developed by Talischi et al. [^1]. This
package allows the generation of polygonal meshes (specifically bounded Voronoi tessellsations)
to a given domain. This is a ```Python``` adaptation of that original work. Here we have altered the
implementation, while remaining faithful to the original intentions of the software, to make the codebase
more pythonic in nature and to fit the user pipline more appropriately.


The functionality of this subpackage is demonstrated in the notebook examples in
the GitHub repository. For more information, see [^2], a presentation given by Pereira et al. at Princeton explaining the 
full behaviour and implementation of the package.

Talischi et al. implemented the code in two dimensions, this is the standard case (and the currently
implemented set-up), see ```two_dimensional``` below for more information on the specifics. The ```two_dimensional```
subpackage contains three modules; ```domains```, ```main``` and ```visualisation```. The ```domains```
module contains predefined domains for use in mesh generation. ```RectangleDomain``` and ```CircleDomain```
for example. One can create custom domains with more structure -- this is a more complicated scenario
and is covered in [^2]. The ```main```module contains the ```poly_mesher``` function itself. Finally, 
```visualisation``` contains a plotting function, with lots of customisation to display the generated
meshes.

Finally, one important note, the cleaning function in the original code has several bugs including a tendency to 
collapse boundary edges. In terms of more recent work (see Calloo et al. [^3]), we require Voronoi tessellations, a 
property of which is not retained by the cleaning functions. There is the potential for small edges but either more or 
less iterations fixes this.

[^1]: Talischi, C., Paulino, G.H., Pereira, A. et al. (2012). PolyMesher: a general-purpose mesh generator for polygonal elements written in Matlab. Struct Multidisc Optim 45, 309–328.
[^2]: Pereira, A., Talischi, C., Menezesand, Ivan F. M., Paulino, G.H., https://paulino.princeton.edu/conferences/presentations/11periera_polymesher.pdf
[^3]: Calloo, A., Evans, M., Lockyer, H., Madiot, F., Pryer, T., & Zanetti, L. (2025). Cycle-free polytopal mesh sweeping for Boltzmann transport. Numerical Algorithms, 1-24.

### Generate XML mesh file for Fenics


In the text below **FILE** depicts the file name 



#### With _netgen_

- In _netgen_ load STL or STEP file (FILE.stl, FILE.step, or FILE.stp)
    - Load for instance from [Example STL](http://forums.reprap.org/read.php?88,6830)
- Generate mesh
- Save in DIFFPACK format (FILE.grid)
- In terminal: $ dolfin-convert FILE.grid FILE.xml

 
#### With _gmsh_

- Create geometry in _gmsh_ and save FILE.msh file 
- In terminal for generation of FILE.msh:
    - (for 2D mesh): $ gmsh -2 FILE.geo 
    - (for 3D mesh): $ gmsh -3 FILE.geo 
- In terminal: $ dolfin-convert FILE.msh FILE.xml


#### Load XML file in _Fenics_

    mesh = Mesh('FILE.xml')


#### References

- [STL example files](http://forums.reprap.org/read.php?88,6830)
- [STL to Fenics](https://en.wikiversity.org/wiki/CAD_to_FEniCS_example)
- [gmsh to Fenics](http://mypages.iit.edu/~asriva13/?page_id=586)

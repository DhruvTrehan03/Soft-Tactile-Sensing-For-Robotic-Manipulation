from netgen.occ import *
from netgen.meshing import MeshingParameters
from ngsolve import Mesh
import netgen.gui  # Import for visualization

# Create a 2D rectangle with width=1 and height=1.25
rectangle = SRectangle(0, 0, 1, 1.25)  # (xmin, ymin, xmax, ymax)

# Generate mesh
geo = OCCGeometry(rectangle)
mesh = geo.GenerateMesh(maxh=0.1)  # Set max element size

# Convert to NGSolve Mesh and visualize
ngmesh = Mesh(mesh)
Draw(ngmesh)

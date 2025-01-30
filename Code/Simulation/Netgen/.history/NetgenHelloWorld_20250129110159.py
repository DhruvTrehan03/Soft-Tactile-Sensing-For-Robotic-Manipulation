from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters
from ngsolve import Mesh
import netgen.gui  # Needed for visualization

# Define the 2D geometry
geo = SplineGeometry()
p1 = geo.AddPoint[0, 0]     # Bottom-left
p2 = geo.AddPoint[1, 0]     # Bottom-right
p3 = geo.AddPoint[1, 1.25]  # Top-right
p4 = geo.AddPoint[0, 1.25]  # Top-left

# Define the rectangle edges
geo.AddSegment(p1, p2)
geo.AddSegment(p2, p3)
geo.AddSegment(p3, p4)
geo.AddSegment(p4, p1)

# Generate and visualize the 2D mesh
mesh = geo.GenerateMesh(maxh=0.1)  # maxh = element size
ngmesh = Mesh(mesh)

Draw(ngmesh)  # Display in Netgen GUI

from netgen.geom2d import SplineGeometry
from netgen.meshing import MeshingParameters
from ngsolve import Mesh,Draw

# Define the 2D geometry
geo = SplineGeometry()
p1 = geo.AppendPoint(0, 0)     # Bottom-left
p2 = geo.AppendPoint(1, 0)     # Bottom-right
p3 = geo.AppendPoint(1, 1.25)  # Top-right
p4 = geo.AppendPoint(0, 1.25)  # Top-left

# Define the rectangle edges
geo.Append (["line", p1, p2])
geo.Append (["line", p2, p3])
geo.Append (["line", p3, p4])
geo.Append (["line", p4, p1])

# Generate and visualize the 2D mesh
mesh = geo.GenerateMesh(maxh=0.05)  # maxh = element size
ngmesh = Mesh(mesh)

Draw(ngmesh)  # Display in Netgen GUI

input("Press Enter to exit...")

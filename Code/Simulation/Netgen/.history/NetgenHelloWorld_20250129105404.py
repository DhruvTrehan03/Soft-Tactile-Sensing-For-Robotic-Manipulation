import netgen.geom2d as geom2d
from ngsolve import *
from ngsolve.webgui import Draw

geo = geom2d.SplineGeometry()
p1 = geo.AppendPoint (0,0)
p2 = geo.AppendPoint (1,0)
p3 = geo.AppendPoint (1,1.25)
p4 = geo.AppendPoint (0,1.25)

geo.Append (["line", p1, p2])
geo.Append (["line", p2, p3])
geo.Append (["line", p3, p4])
geo.Append (["line", p4, p1])

mesh = geo.GenerateMesh (maxh=0.1)

Draw(mesh)
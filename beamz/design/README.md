# Design

+ core.py       / Main module to define and organiz complex design geometries, materials, ...
+ structures.py / Polygon objects to define geometry within the design
+ materials.py  / Material response implemenations (Sellmeier, Drude, etc.)
+ library.py    / Instances of materials with exp. measurements for various materials (Si, SiO, InP, ...)
+ meshing.py    / Turns parametric design into rasterized grid.
+ io.py         / Import and export of the design as .gds, .gltf, etc.


## core.py
class Design()
    def __init__()              // Initialize the design
    def __str__()               // Print out object info
    def __iadd__()              // Simplified add op
    --------------------------------------------------------------------------
    def _init_boundaries()      // Initialize standard boundary conditions
    def _determin_if_3d()       // 
    def _simplify_poly_for_3d()
    def _unify_polygon()
    --------------------------------------------------------------------------
    def get_material_value()
    --------------------------------------------------------------------------
    def add()
    def scatter()
    def make_grid()             // <- meshing.py
    def import_gds()            // <- io.py
    def export_gds()            // <- io.py
    def show()                  // <- visual/plot.py
    def copy()                  // Create a new copy of this object

## io.py
def import_gds()
def export_gds()

## structures.py
class Polygon()
class Rectangle(Polygon)
class Circle(Polygon)
class Ring(Polygon)
class CircularBend(Polygon)
class Taper(Polygon)

## materials.py
class Matrial()
class CustomMaterial()

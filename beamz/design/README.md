# Design

Module to define the design of complex structures parametrically as well as mesh it into grids of material values.

+ core.py       / Main module to define and organiz complex design geometries, materials, ...
+ structures.py / Polygon objects to define geometry within the design
+ materials.py  / Material response implemenations (Sellmeier, Drude, etc.)
+ library.py    / Instances of materials with exp. measurements for various materials (Si, SiO, InP, ...)
+ meshing.py    / Turns parametric design into rasterized grid.
+ io.py         / Import and export of the design as .gds, .gltf, etc.
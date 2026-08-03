from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pytest

import beamz
import beamz.design.raster as raster
from beamz.design.raster.importers import from_gdsfactory


class FakeComponent:
    def get_polygons_points(self, by):
        assert by == "tuple"
        return {(1, 0): [np.array([[0, 0], [2, 0], [2, 1], [0, 1]])]}


@dataclass
class FakeLayer:
    layer: tuple[int, int] = (1, 0)
    zmin: float = 0.0
    thickness: float = 0.22
    material: str = "si"
    sidewall_angle: float = 0.0
    width_to_z: float = 0.0
    z_to_bias: object | None = None
    mesh_order: int = 0
    bias: float = 0.0


@dataclass
class FakeStack:
    layers: dict


def test_material_defaults_require_explicit_opt_in():
    stack = FakeStack({"core": FakeLayer()})
    with pytest.raises(ValueError, match="material_map"):
        from_gdsfactory(FakeComponent(), stack)

    scene = from_gdsfactory(
        FakeComponent(), stack, material_map={"si": beamz.Material(permittivity=12)}
    )
    assert scene.materials[1].epsilon_r[:3] == (12.0, 12.0, 12.0)
    raster.compile_scene(scene)


def test_callable_z_bias_lowers_to_supported_tapered_segments():
    stack = FakeStack(
        {
            "core": FakeLayer(
                z_to_bias=lambda z: 0.05 * z * z,
            )
        }
    )
    scene = from_gdsfactory(
        FakeComponent(), stack, material_map={"si": raster.Material(12)}
    )

    assert len(scene.objects) > 1
    assert all(
        isinstance(
            obj.geometry,
            (raster.ExtrudedPolygon, raster.TaperedExtrudedPolygon),
        )
        for obj in scene.objects
    )
    raster.compile_scene(scene)

    segments = sorted(scene.objects, key=lambda object_: object_.geometry.z_min)
    for segment in segments:
        geometry = segment.geometry
        z0 = geometry.z_min / (0.22e-6)
        z1 = geometry.z_max / (0.22e-6)
        low_bias = -min(x for x, _ in geometry.polygon.exterior)
        high_bias = low_bias
        if isinstance(geometry, raster.TaperedExtrudedPolygon):
            high_bias -= (geometry.z_max - geometry.z_min) * np.tan(
                np.radians(geometry.sidewall_angle_degrees)
            )
        np.testing.assert_allclose(low_bias, 0.05 * z0 * z0 * 1e-6, atol=1e-9)
        np.testing.assert_allclose(high_bias, 0.05 * z1 * z1 * 1e-6, atol=1e-9)


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore::ResourceWarning")
def test_real_gdsfactory_active_pdk_import_and_rasterization():
    gf = pytest.importorskip("gdsfactory")
    # GDSFactory 8 emits Pydantic-v2 migration warnings while constructing its
    # generic PDK; version 9 requires that PDK to be activated explicitly.
    # Both are upstream lifecycle details, not importer output.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            pdk = gf.get_active_pdk()
        except ValueError:
            from gdsfactory.gpdk import PDK

            PDK.activate()
            pdk = gf.get_active_pdk()
        component = gf.components.straight(length=2.0, width=0.5)
        layer_stack = pdk.layer_stack
    material_names = sorted(
        {
            str(level.material)
            for level in layer_stack.layers.values()
            if getattr(level, "material", None)
        }
    )
    material_map = {
        name: raster.Material(2.0 + index) for index, name in enumerate(material_names)
    }

    scene = from_gdsfactory(component, layer_stack, material_map=material_map)
    grid = raster.Grid.uniform(
        (-1e-6, -1e-6, -1e-6),
        (3e-6, 1e-6, 2e-6),
        (8, 4, 3),
    )
    first = scene.rasterize(grid, options=raster.RasterOptions(quality="fast"))
    second = scene.rasterize(grid, options=raster.RasterOptions(quality="fast"))

    assert scene.objects
    assert float(first.tensors["epsilon"].max()) > 1.0
    np.testing.assert_array_equal(first.tensors["epsilon"], second.tensors["epsilon"])

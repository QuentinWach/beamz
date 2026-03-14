from zview import CameraSpec, ClipPlaneSpec, MaterialSpec, Object3D, SceneSpec, scene_from_dict


def test_scene_round_trip_preserves_structure():
    scene = SceneSpec(
        title="Example",
        objects=[
            Object3D(
                kind="box",
                label="Domain",
                geometry={"center": [0.0, 0.0, 0.0], "size": [1.0, 2.0, 3.0]},
                material=MaterialSpec(color="#123456", opacity=0.2, wireframe=True),
            )
        ],
        clip_planes=[ClipPlaneSpec(normal=(0.0, 0.0, 1.0), constant=0.0, enabled=False)],
        camera=CameraSpec(position=(3.0, 2.0, 1.0), target=(0.0, 0.0, 0.0)),
    )

    payload = scene.to_dict()
    restored = scene_from_dict(payload)

    assert restored.title == "Example"
    assert restored.objects[0].kind == "box"
    assert restored.objects[0].material.wireframe is True
    assert restored.camera.position == (3.0, 2.0, 1.0)
    assert restored.clip_planes[0].enabled is False

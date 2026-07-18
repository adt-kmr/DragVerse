# Unity batch-mode twin generator

Consumes the `scene.json` written by `twin/generator.py` (`POST /generate-twin`) and
produces a baked Unity scene.

```bash
Unity -batchmode -quit -nographics \
      -projectPath twin/unity_project \
      -executeMethod TwinForge.TwinGenerator.Build \
      -scene ../../data/twins/<mesh_id>/scene.json \
      -out Assets/Generated/Twin.unity
```

Prefabs are looked up by the `prefab` path in `twin/rules/mapping.yaml` under any
`Assets/Resources/` folder (e.g. `Assets/Resources/Furniture/RectTable.prefab`). None ship
with the repo — an unresolved prefab falls back to a correctly sized primitive, so the bake
and the collider geometry the planner depends on still come out right.

The `navmesh.json` next to `scene.json` is the Python planner's occupancy grid; Unity bakes
its own navmesh from these colliders.

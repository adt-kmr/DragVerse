using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace TwinForge
{
    /// <summary>
    /// Batch-mode consumer of the scene manifest written by twin/generator.py.
    ///
    /// Positions and collider sizes arrive already converted to Unity's left-handed
    /// Y-up frame (see twin/generator.py UNITY_FROM_WORLD) — nothing is re-mapped here.
    /// The navmesh grid that ships next to scene.json is for the Python planner;
    /// Unity bakes its own from the colliders this script adds.
    ///
    /// Run:
    ///   Unity -batchmode -quit -projectPath twin/unity_project \
    ///         -executeMethod TwinForge.TwinGenerator.Build \
    ///         -scene data/twins/&lt;mesh_id&gt;/scene.json -out Assets/Generated/Twin.unity
    /// </summary>
    public static class TwinGenerator
    {
        [Serializable]
        public class ColliderSpec
        {
            public string type;
            public float[] size;
        }

        [Serializable]
        public class SceneObject
        {
            public string id;
            public string label;
            public string prefab;
            public float[] position;
            public ColliderSpec collider;
            public float confidence;
        }

        [Serializable]
        public class SceneManifest
        {
            public SceneObject[] objects;
        }

        public static void Build()
        {
            string manifestPath = Arg("-scene") ?? "scene.json";
            string outPath = Arg("-out") ?? "Assets/Generated/Twin.unity";

            var manifest = JsonUtility.FromJson<SceneManifest>(File.ReadAllText(manifestPath));
            if (manifest?.objects == null)
            {
                throw new Exception($"no objects in scene manifest {manifestPath}");
            }

            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects,
                                                    NewSceneMode.Single);
            var root = new GameObject("Twin");
            foreach (var obj in manifest.objects)
            {
                Spawn(obj).transform.SetParent(root.transform, true);
            }

            // Legacy bake: everything static so the walkable surface comes from the
            // colliders above, matching what the Python navmesh grid blocks.
            foreach (Transform child in root.transform)
            {
                GameObjectUtility.SetStaticEditorFlags(
                    child.gameObject, StaticEditorFlags.NavigationStatic);
            }
            UnityEditor.AI.NavMeshBuilder.BuildNavMesh();

            Directory.CreateDirectory(Path.GetDirectoryName(outPath));
            EditorSceneManager.SaveScene(scene, outPath);
            Debug.Log($"TwinForge: wrote {outPath} with {manifest.objects.Length} objects");
        }

        private static GameObject Spawn(SceneObject obj)
        {
            // A missing prefab must not fail the bake — a sized primitive still gives the
            // planner a correct collider, which is what the pipeline actually consumes.
            var source = Resources.Load<GameObject>(obj.prefab);
            var go = source != null
                ? UnityEngine.Object.Instantiate(source)
                : GameObject.CreatePrimitive(PrimitiveType.Cube);

            go.name = $"{obj.label}_{obj.id}";
            go.transform.position = Vec(obj.position);

            var size = Vec(obj.collider?.size);
            if (source == null)
            {
                // The primitive's own 1m cube collider is the right shape once scaled.
                go.transform.localScale = size;
            }
            else
            {
                AddCollider(go, obj.collider?.type, size);
            }
            return go;
        }

        private static void AddCollider(GameObject go, string type, Vector3 size)
        {
            switch (type)
            {
                case "CapsuleCollider":
                    var capsule = go.AddComponent<CapsuleCollider>();
                    capsule.height = size.y;
                    capsule.radius = Mathf.Max(size.x, size.z) / 2f;
                    break;
                case "MeshCollider":
                    if (go.GetComponentInChildren<MeshFilter>() != null)
                    {
                        go.AddComponent<MeshCollider>().convex = true;
                        break;
                    }
                    // No mesh to wrap — a box is better than no collider at all.
                    goto default;
                default:
                    go.AddComponent<BoxCollider>().size = size;
                    break;
            }
        }

        private static Vector3 Vec(float[] v)
        {
            return v != null && v.Length >= 3 ? new Vector3(v[0], v[1], v[2]) : Vector3.one;
        }

        private static string Arg(string name)
        {
            var args = Environment.GetCommandLineArgs();
            for (int i = 0; i < args.Length - 1; i++)
            {
                if (args[i] == name)
                {
                    return args[i + 1];
                }
            }
            return null;
        }
    }
}

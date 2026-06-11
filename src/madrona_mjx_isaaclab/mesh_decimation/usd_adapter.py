from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MeshArrays:
    points: np.ndarray
    face_counts: np.ndarray
    face_indices: np.ndarray
    uvs: np.ndarray | None
    uv_interpolation: str | None


def is_collision_prim(prim) -> bool:
    path_str = prim.GetPrimPath().pathString
    if "/collisions/" in path_str or "/Collision" in path_str:
        return True
    try:
        from pxr import UsdPhysics
        if prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            return True
    except Exception:
        pass
    return False


def read_mesh_arrays(mesh) -> MeshArrays | None:
    from pxr import UsdGeom

    points_attr = mesh.GetPointsAttr()
    face_counts_attr = mesh.GetFaceVertexCountsAttr()
    face_indices_attr = mesh.GetFaceVertexIndicesAttr()

    if not (points_attr.HasValue() and face_counts_attr.HasValue() and face_indices_attr.HasValue()):
        return None

    points = np.array(points_attr.Get(), dtype=np.float32)
    face_counts = np.array(face_counts_attr.Get(), dtype=np.int32)
    face_indices = np.array(face_indices_attr.Get(), dtype=np.int32)

    uvs = None
    uv_interpolation = None
    prim = mesh.GetPrim()
    primvars_api = UsdGeom.PrimvarsAPI(prim)
    for uv_name in ("st", "UVMap", "uv"):
        pv = primvars_api.GetPrimvar(uv_name)
        if pv and pv.HasValue():
            uvs = np.array(pv.Get(), dtype=np.float32)
            uv_interpolation = pv.GetInterpolation()
            break

    return MeshArrays(
        points=points,
        face_counts=face_counts,
        face_indices=face_indices,
        uvs=uvs,
        uv_interpolation=uv_interpolation,
    )


def write_mesh_arrays(mesh, points: np.ndarray, triangles: np.ndarray, uvs: np.ndarray | None = None) -> None:
    from pxr import Vt

    mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.GetFaceVertexCountsAttr().Set([3] * len(triangles))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles.flatten().astype(np.int32)))

    if uvs is not None:
        from pxr import UsdGeom
        prim = mesh.GetPrim()
        primvars_api = UsdGeom.PrimvarsAPI(prim)
        for uv_name in ("st", "UVMap", "uv"):
            pv = primvars_api.GetPrimvar(uv_name)
            if pv and pv.HasValue():
                pv.Set(Vt.Vec2fArray.FromNumpy(uvs.astype(np.float32)))
                pv.SetInterpolation("vertex")
                break

    clear_geometry_dependent_primvars(mesh)


def clear_geometry_dependent_primvars(mesh) -> None:
    from pxr import UsdGeom

    prim = mesh.GetPrim()
    stage = prim.GetStage()

    normals_attr = mesh.GetNormalsAttr()
    if normals_attr and normals_attr.HasAuthoredValue():
        normals_attr.Clear()

    # UV primvars are explicitly re-authored by write_mesh_arrays with the new
    # vertex layout. Skipping them here so we don't immediately Block() the value
    # we just wrote -- that's what was making textured meshes render grey.
    UV_NAMES = ("st", "UVMap", "uv")
    api = UsdGeom.PrimvarsAPI(prim)
    for pv in api.GetPrimvars():
        if pv.GetPrimvarName() in UV_NAMES:
            continue
        interp = pv.GetInterpolation()
        if interp in ("vertex", "varying", "faceVarying", "uniform"):
            pv.GetAttr().Block()

    subset_paths = [c.GetPath() for c in prim.GetAllChildren() if c.IsA(UsdGeom.Subset)]
    for path in subset_paths:
        stage.RemovePrim(path)

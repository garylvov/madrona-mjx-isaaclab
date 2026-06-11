from typing import NamedTuple

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


class StaticMeshRecord(NamedTuple):
    """A UsdGeom.Mesh prim together with its world-space pose at time 0."""

    prim: Usd.Prim
    pos_world: np.ndarray
    quat_world_wxyz: np.ndarray
    scale_world: np.ndarray


def _is_descendant_or_equal(path: Sdf.Path, covered: set[Sdf.Path]) -> bool:
    # Linear scan — expected len(covered) is small (tens, not thousands).
    for cp in covered:
        if path == cp or path.HasPrefix(cp):
            return True
    return False


def _decompose_xform(mat: Gf.Matrix4d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation = np.array(mat.ExtractTranslation(), dtype=np.float64)

    r = mat.ExtractRotationMatrix()
    # Columns of the 3x3 rotation-scale block (GfMatrix3d is row-major: r[row][col])
    col0 = np.array([r[0][0], r[1][0], r[2][0]], dtype=np.float64)
    col1 = np.array([r[0][1], r[1][1], r[2][1]], dtype=np.float64)
    col2 = np.array([r[0][2], r[1][2], r[2][2]], dtype=np.float64)

    sx = float(np.linalg.norm(col0))
    sy = float(np.linalg.norm(col1))
    sz = float(np.linalg.norm(col2))
    scale = np.array([sx, sy, sz], dtype=np.float64)

    eps = 1e-12
    sx = sx if sx > eps else 1.0
    sy = sy if sy > eps else 1.0
    sz = sz if sz > eps else 1.0

    # Build a pure-rotation Matrix4d, then extract its quaternion via Gf.
    norm4 = Gf.Matrix4d(
        col0[0] / sx, col0[1] / sx, col0[2] / sx, 0.0,
        col1[0] / sy, col1[1] / sy, col1[2] / sy, 0.0,
        col2[0] / sz, col2[1] / sz, col2[2] / sz, 0.0,
        0.0,          0.0,          0.0,           1.0,
    )
    q = norm4.ExtractRotationQuat()
    imag = q.GetImaginary()
    quat_wxyz = np.array([q.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)
    return translation, quat_wxyz, scale


def walk_static_meshes(
    stage: Usd.Stage,
    env_root_path: str,
    covered_paths: set[Sdf.Path],
    *,
    max_meshes: int = 5000,
    skip_purposes: tuple[str, ...] = ("guide", "proxy"),
) -> list[StaticMeshRecord]:
    """Walk `env_root_path` subtree of `stage` and return every UsdGeom.Mesh prim
    suitable for static ingestion into Madrona.

    Skip conditions (prune subtree, do not descend):
      - prim path is in `covered_paths` (or is a descendant of any path in covered_paths)
      - UsdGeom.Imageable visibility is "invisible"
      - UsdGeom.Imageable purpose is in `skip_purposes`
      - prim has UsdPhysics.ArticulationRootAPI applied
    Within a non-pruned subtree, any UsdGeom.Mesh prim is recorded.
    Records are returned in stage traversal order.

    Pose math: pos/quat come from UsdGeom.Xformable.ComputeLocalToWorldTransform(0)
    decomposed into translation + rotation + scale.
    """
    root_prim = stage.GetPrimAtPath(env_root_path)
    if not root_prim.IsValid():
        raise ValueError(f"No prim at path: {env_root_path!r}")

    time = Usd.TimeCode.Default()
    records: list[StaticMeshRecord] = []

    prim_range = iter(Usd.PrimRange(root_prim))
    for prim in prim_range:
        path = prim.GetPath()

        if _is_descendant_or_equal(path, covered_paths):
            prim_range.PruneChildren()
            continue

        imageable = UsdGeom.Imageable(prim)
        if imageable:
            visibility = imageable.ComputeVisibility(time)
            if visibility == UsdGeom.Tokens.invisible:
                prim_range.PruneChildren()
                continue

            purpose = imageable.ComputePurpose()
            if purpose in skip_purposes:
                prim_range.PruneChildren()
                continue

        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim_range.PruneChildren()
            continue

        if prim.IsA(UsdGeom.Mesh):
            xformable = UsdGeom.Xformable(prim)
            mat = xformable.ComputeLocalToWorldTransform(time)
            pos, quat_wxyz, scale = _decompose_xform(mat)

            if len(records) >= max_meshes:
                raise RuntimeError(
                    f"walk_static_meshes: exceeded max_meshes={max_meshes} "
                    f"while traversing {env_root_path!r}. "
                    f"Increase max_meshes or narrow env_root_path."
                )

            records.append(StaticMeshRecord(
                prim=prim,
                pos_world=pos,
                quat_world_wxyz=quat_wxyz,
                scale_world=scale,
            ))

    return records

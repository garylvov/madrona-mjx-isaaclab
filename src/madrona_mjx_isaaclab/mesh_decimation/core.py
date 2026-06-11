from __future__ import annotations

import ctypes
import importlib.metadata
import logging

import numpy as np

import meshoptimizer as mo
import meshoptimizer._loader as _mo_loader

_log = logging.getLogger(__name__)

# drop this guard once meshoptimizer fixes argtypes upstream
_PATCHED_VERSION = "0.2.30a0"
_mo_version = importlib.metadata.version("meshoptimizer")
if _mo_version == _PATCHED_VERSION:
    # meshoptimizer 0.2.30a0 passes target_index_count as a plain Python int,
    # but ctypes requires c_size_t for that parameter. Setting argtypes coerces correctly.
    _mo_loader.lib.meshopt_simplifyWithAttributes.argtypes = [
        ctypes.POINTER(ctypes.c_uint),
        ctypes.POINTER(ctypes.c_uint),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_ubyte),
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
else:
    _log.warning(
        "meshoptimizer version %s is not the known-patched version %s; "
        "skipping ctypes argtypes patch for meshopt_simplifyWithAttributes",
        _mo_version,
        _PATCHED_VERSION,
    )


def triangulate(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    triangles = []
    idx = 0
    for count in face_counts:
        for i in range(1, count - 1):
            triangles.append([face_indices[idx], face_indices[idx + i], face_indices[idx + i + 1]])
        idx += count
    return np.array(triangles, dtype=np.int32)


def expand_face_varying(
    points: np.ndarray,
    face_counts: np.ndarray,
    face_indices: np.ndarray,
    uvs_fv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_vertices = []
    new_uvs = []
    new_faces = []
    idx = 0
    for count in face_counts:
        base = len(new_vertices)
        for i in range(count):
            new_vertices.append(points[face_indices[idx + i]])
            new_uvs.append(uvs_fv[idx + i])
        for i in range(1, count - 1):
            new_faces.append([base, base + i, base + i + 1])
        idx += count
    return (
        np.array(new_vertices, dtype=np.float32),
        np.array(new_uvs, dtype=np.float32),
        np.array(new_faces, dtype=np.int32),
    )


def _resolve_target_index_count(triangles: np.ndarray, spec) -> int:
    ratio_count = int(triangles.size * spec.target_ratio)
    if spec.target_vertex_count is not None:
        # closed manifold: indices ~= 6 * vertices (each vert touches ~6 tris on average)
        vertex_count_idx = spec.target_vertex_count * 6
        return min(ratio_count, vertex_count_idx)
    return ratio_count


def weld_by_attributes(
    points: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray | None = None,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Merge coincident vertices that share BOTH position and (if present) UV.

    CAD exporters routinely split every face for hard-edge normals, producing
    vertex soups: same 3D position but separate vertex records (and no shared
    edges, which structurally blocks meshoptimizer's QEM simplifier). This
    weld merges duplicates that match across all attributes, while preserving
    UV-seam pairs (same position, different UV) as separate vertices.

    Returns (welded_points, welded_triangles, welded_uvs_or_None).
    """
    if len(points) == 0:
        return points, triangles, uvs

    if uvs is None:
        stream = points
    else:
        stream = np.concatenate([points, uvs], axis=1)

    quantized = np.round(stream / eps).astype(np.int64)
    _, unique_idx, inverse = np.unique(
        quantized, axis=0, return_index=True, return_inverse=True
    )
    new_points = points[unique_idx]
    new_uvs = uvs[unique_idx] if uvs is not None else None
    flat_idx = triangles.reshape(-1).astype(np.int64)
    new_triangles = inverse[flat_idx].reshape(triangles.shape).astype(triangles.dtype)
    return new_points, new_triangles, new_uvs


def _compact(points: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used = np.unique(indices)
    remap = np.zeros(len(points), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return points[used], remap[indices]


def _compact_with_attrs(
    points: np.ndarray,
    uvs: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    used = np.unique(indices)
    remap = np.zeros(len(points), dtype=np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    return points[used], uvs[used], remap[indices]


def simplify_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    spec,
    uvs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    # Weld coincident (pos, UV) duplicates first. Required because vertex-soup
    # source meshes (CAD exporter style) have no shared edges; QEM cannot
    # collapse them. UV seams are preserved (same pos, different UV stay separate).
    points, triangles, uvs = weld_by_attributes(points, triangles, uvs)

    target_idx_count = _resolve_target_index_count(triangles, spec)
    if target_idx_count >= triangles.size:
        return points, triangles, uvs

    flat_indices = triangles.flatten().astype(np.uint32)
    pts_f32 = points.astype(np.float32)

    if uvs is None:
        dest = np.empty(len(flat_indices), dtype=np.uint32)
        n = mo.simplify(
            dest,
            flat_indices,
            pts_f32,
            target_index_count=target_idx_count,
            target_error=spec.target_error,
        )
        new_idx = dest[:n].astype(np.int32)
        new_pts, new_idx = _compact(pts_f32, new_idx)
        return new_pts, new_idx.reshape(-1, 3), None

    attrs = uvs.astype(np.float32)
    weights = np.array([spec.uv_weight, spec.uv_weight], dtype=np.float32)
    dest = np.empty(len(flat_indices), dtype=np.uint32)
    n = mo.simplify_with_attributes(
        dest,
        flat_indices,
        pts_f32,
        attrs,
        weights,
        target_index_count=target_idx_count,
        target_error=spec.target_error_uv,
    )
    new_idx = dest[:n].astype(np.int32)
    new_pts, new_uvs, new_idx = _compact_with_attrs(pts_f32, attrs, new_idx)
    return new_pts, new_idx.reshape(-1, 3), new_uvs

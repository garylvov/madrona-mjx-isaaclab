from __future__ import annotations

import os
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MeshDecimateSpec:
    target_ratio: float
    target_vertex_count: int | None
    min_faces: int
    target_error: float
    target_error_uv: float
    uv_weight: float
    bake_textures: bool
    expand_face_varying_uvs: bool = True

    @classmethod
    def from_level(cls, level: str, **overrides) -> "MeshDecimateSpec":
        return spec_from_level(level, **overrides)


BAKE_LEVELS: dict[str, dict] = {
    "light": {
        "target_ratio": 0.7,
        "min_faces": 64,
        "bake_textures": True,
        "target_error": 0.02,
        "target_error_uv": 0.05,
        "decimate_default": 0.3,
        "decimate_keep_ratio": 0.6,
        "decimate_robot_ratio": 0.8,
        "decimate_min_faces": 20,
        "bake_max_texture_size": 1024,
    },
    "medium": {
        "target_ratio": 0.4,
        "min_faces": 64,
        "bake_textures": True,
        "target_error": 0.05,
        "target_error_uv": 0.1,
        "decimate_default": 0.1,
        "decimate_keep_ratio": 0.5,
        "decimate_robot_ratio": 0.7,
        "decimate_min_faces": 20,
        "bake_max_texture_size": 512,
    },
    "high": {
        "target_ratio": 0.2,
        "min_faces": 32,
        "bake_textures": True,
        "target_error": 0.1,
        "target_error_uv": 0.2,
        "decimate_default": 0.05,
        "decimate_keep_ratio": 0.3,
        "decimate_robot_ratio": 0.5,
        "decimate_min_faces": 20,
        "bake_max_texture_size": 256,
    },
}

_DEFAULTS = {
    "target_ratio": 0.4,
    "target_vertex_count": None,
    "min_faces": 64,
    "target_error": 1e-2,
    "target_error_uv": 0.1,
    "uv_weight": 0.5,
    "bake_textures": True,
    "expand_face_varying_uvs": True,
}


_SPEC_FIELDS = frozenset(_DEFAULTS)


def spec_from_level(level: str, **overrides) -> MeshDecimateSpec:
    if level not in BAKE_LEVELS:
        raise KeyError(f"Unknown bake level {level!r}. Valid: {list(BAKE_LEVELS)}")
    merged = {**_DEFAULTS, **BAKE_LEVELS[level], **overrides}
    kwargs = {k: v for k, v in merged.items() if k in _SPEC_FIELDS}
    return MeshDecimateSpec(**kwargs)


def spec_from_env(base: MeshDecimateSpec) -> MeshDecimateSpec:
    changes: dict = {}

    raw_target = os.environ.get("MADRONA_TARGET_VERTICES")
    raw_max = os.environ.get("MADRONA_MAX_VERTICES")

    if raw_target is not None and raw_max is not None:
        target = min(int(raw_target), int(raw_max))
        changes["target_vertex_count"] = target
    elif raw_target is not None:
        changes["target_vertex_count"] = int(raw_target)
    elif raw_max is not None:
        changes["target_vertex_count"] = int(raw_max)

    if not changes:
        return base
    return replace(base, **changes)

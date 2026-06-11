# Copyright (c) 2025, Gigastrap Authors
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Copyright (c) 2025, Gigastrap Authors
# SPDX-License-Identifier: BSD-3-Clause

"""Scene adapter for converting Isaac Lab state to Madrona format.

VERSION: 2025-12-03-v4 (reverted scale fix, added docs)

Handles:
- Extracting geometry from USD scene (one-time at init)
- Syncing transforms from PhysX tensors to Madrona state (per-frame)
- Extracting visualization markers (goal cubes, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.utils.math import convert_camera_frame_orientation_convention


# JIT-compiled kernels for state building (fuses tensor ops, reduces kernel launch overhead)
@torch.jit.script
def _jit_scatter_poses(
    geom_positions_out: torch.Tensor,
    geom_quaternions_out: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    geom_indices: torch.Tensor,
    env_origins_expanded: torch.Tensor,
) -> None:
    """JIT kernel for scattering body poses to geometry output."""
    geom_positions_out[:, geom_indices, :] = pos - env_origins_expanded
    geom_quaternions_out[:, geom_indices, :] = quat


@torch.jit.script
def _jit_quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Inline quat_mul for JIT compilation (avoids cross-module call)."""
    # Quaternion multiplication: q1 * q2 (wxyz format)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


@torch.jit.script
def _jit_camera_transform(
    cam_pos_out: torch.Tensor,
    cam_quat_out: torch.Tensor,
    camera_positions: torch.Tensor,
    quat_opengl_flat: torch.Tensor,
    env_origins_expanded: torch.Tensor,
    to_y_fwd: torch.Tensor,
    num_envs: int,
    num_cams: int,
) -> None:
    """JIT kernel for camera position/orientation transform."""
    cam_pos_out[:] = camera_positions - env_origins_expanded
    quat_madrona = _jit_quat_mul(quat_opengl_flat, to_y_fwd)
    cam_quat_out[:] = quat_madrona.view(num_envs, num_cams, 4)


from .geometry import (
    GeometryAdapterFactory,
    MadronaGeometry,
    SingleGeometry,
    combine_geometries,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class MadronaState:
    """Per-frame state for Madrona rendering."""

    geom_positions: torch.Tensor  # (num_worlds, num_geoms, 3)
    geom_quaternions: (
        torch.Tensor
    )  # (num_worlds, num_geoms, 4) wxyz (Madrona uses MuJoCo convention)
    geom_scales: torch.Tensor  # (num_worlds, num_geoms, 3)
    camera_positions: torch.Tensor  # (num_worlds, num_cameras, 3)
    camera_quaternions: (
        torch.Tensor
    )  # (num_worlds, num_cameras, 4) wxyz (Madrona uses MuJoCo convention)
    light_positions: torch.Tensor  # (num_worlds, num_lights, 3)
    light_directions: torch.Tensor  # (num_worlds, num_lights, 3)


@dataclass
class LightInfo:
    """Information about a light in the scene."""

    light_type: int  # 0=directional, 1=point, 2=spot
    position: np.ndarray  # (3,) position
    direction: np.ndarray  # (3,) direction (for directional/spot)
    color: np.ndarray  # (3,) RGB color
    intensity: float
    casts_shadow: bool


@dataclass
class GeomMapping:
    """Maps a Madrona geometry index to Isaac Lab asset."""

    asset_type: str  # "articulation", "rigid_object", or "marker"
    asset_name: str
    body_index: (
        int  # For articulations, which body; for rigid objects/markers, always 0
    )
    marker_instance_index: int | None = (
        None  # For markers, which instance in the PointInstancer
    )


class SceneToMadronaAdapter:
    """Adapts Isaac Lab scene to Madrona's expected format.

    Supports two modes:
    - shared_geometry=True (default): All envs share same geometry from env_0
    - shared_geometry=False: Extract geometry per environment (for varied objects)
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        shared_geometry: bool = True,
        ingest_stage_meshes: bool = True,
        max_texture_size: int | None = None,
    ):
        self._env = env
        self._device = env.device
        self._num_envs = env.num_envs
        self._shared_geometry = shared_geometry
        self._ingest_stage_meshes = ingest_stage_meshes
        # Texture max-edge for every texture this renderer ingests; all cameras
        # of the renderer share the geometry/texture pool, so this is per
        # renderer instance (set from the camera config; None falls back to the
        # MADRONA_MAX_TEXTURE_SIZE env var, then the default).
        self._max_texture_size = max_texture_size

        self._geom_mappings: list[GeomMapping] = []
        self._geometry: MadronaGeometry | None = None
        # Per-env geometry for non-shared mode
        self._per_env_geometries: list[MadronaGeometry] | None = None
        self._per_env_geom_mappings: list[list[GeomMapping]] | None = None
        self._lights: list[LightInfo] = []
        self._marker_prim_paths: list[str] = []
        self._lights_logged = False
        self._asset_geometry_templates: dict[str, list[SingleGeometry]] = {}

        self._articulation_body_counts: dict[str, int] = {}

        # Optimized lookup structures (built after extract_geometry)
        self._geom_lookup_built = False
        self._articulation_geom_indices: list[int] = []
        self._articulation_asset_refs: list = (
            []
        )  # Direct references to articulation objects
        self._articulation_body_indices: list[int] = []
        self._rigid_geom_indices: list[int] = []
        self._rigid_asset_refs: list = []  # Direct references to rigid objects
        # Init-time scale per rigid_object (asset_name -> (3,) float32 ndarray).
        # Populated by _extract_rigid_object_geometry; used in _build_geom_lookup_tables
        # to build the per-frame geom_scales_out scatter tensor.
        self._rigid_asset_scales: dict[str, np.ndarray] = {}
        self._marker_geom_indices: list[int] = []
        self._marker_asset_names: list[str] = []

        # Static USD mesh tracking (env_idx -> list of (pos_world, quat_wxyz) per record)
        # Poses are stored in world-frame; env-local subtraction happens at scatter time.
        self._static_geom_local_poses: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}

        # Pre-computed tensors for per-frame static scatter (built in _build_geom_lookup_tables)
        self._static_usd_geom_indices: torch.Tensor | None = None
        self._static_usd_positions: torch.Tensor | None = None   # (num_envs, num_static, 3) env-local
        self._static_usd_quats: torch.Tensor | None = None        # (num_envs, num_static, 4) wxyz

        # Cached static tensors
        self._to_y_fwd: torch.Tensor | None = None
        self._to_y_fwd_expanded: torch.Tensor | None = None
        self._light_positions_cached: torch.Tensor | None = None
        self._light_directions_cached: torch.Tensor | None = None
        self._identity_quat: torch.Tensor | None = None

    def _scene_rigid_objects(self):
        """Return scene rigid objects when the backend supports them."""
        try:
            return self._env.scene.rigid_objects
        except NotImplementedError:
            return {}

    @staticmethod
    def _copy_geometry(geom: SingleGeometry) -> SingleGeometry:
        """Return a deep-enough copy for adding another geom instance."""
        return SingleGeometry(
            vertices=geom.vertices.copy(),
            normals=geom.normals.copy(),
            indices=geom.indices.copy(),
            uvs=geom.uvs.copy() if geom.uvs is not None else None,
            color=geom.color.copy(),
            texture=geom.texture,
        )

    @property
    def num_geoms(self) -> int:
        return len(self._geom_mappings)

    @property
    def num_lights(self) -> int:
        return len(self._lights)

    @property
    def geometry(self) -> MadronaGeometry:
        if self._geometry is None:
            raise RuntimeError("Call extract_geometry() first")
        return self._geometry

    @property
    def lights(self) -> list[LightInfo]:
        return self._lights

    def _build_geom_lookup_tables(self):
        """Build optimized lookup structures for fast per-frame state updates.

        Called once after extract_geometry(). Replaces per-frame dict lookups
        with direct array indexing for ~10-20% speedup.
        """
        if self._geom_lookup_built:
            return

        self._articulation_geom_indices = []
        self._articulation_asset_refs = []
        self._articulation_body_indices = []
        self._rigid_geom_indices = []
        self._rigid_asset_refs = []
        self._rigid_geom_scale_list: list[np.ndarray] = []
        self._marker_geom_indices = []
        self._marker_asset_names = []

        # Per-asset slot maps for clients (e.g. goal-pass renderer) that need to
        # selectively rewrite a subset of geom slots.
        # _asset_geom_ranges:        asset_name -> list[geom_idx]
        # _asset_body_indices:       asset_name -> list[physx body_idx] (articulations only,
        #                            aligned with _asset_geom_ranges[name])
        self._asset_geom_ranges: dict[str, list[int]] = {}
        self._asset_body_indices: dict[str, list[int]] = {}

        for geom_idx, mapping in enumerate(self._geom_mappings):
            self._asset_geom_ranges.setdefault(mapping.asset_name, []).append(geom_idx)
            if mapping.asset_type == "articulation":
                self._articulation_geom_indices.append(geom_idx)
                self._articulation_asset_refs.append(
                    self._env.scene.articulations[mapping.asset_name]
                )
                self._articulation_body_indices.append(mapping.body_index)
                self._asset_body_indices.setdefault(mapping.asset_name, []).append(
                    mapping.body_index
                )
            elif mapping.asset_type == "rigid_object":
                self._rigid_geom_indices.append(geom_idx)
                self._rigid_asset_refs.append(
                    self._scene_rigid_objects()[mapping.asset_name]
                )
                self._rigid_geom_scale_list.append(
                    self._rigid_asset_scales.get(
                        mapping.asset_name, np.array([1.0, 1.0, 1.0], dtype=np.float32)
                    )
                )
            elif mapping.asset_type == "marker":
                self._marker_geom_indices.append(geom_idx)
                self._marker_asset_names.append(mapping.asset_name)

        # Pre-compute static_usd scatter tensors (constant per-frame, env-local positions)
        self._build_static_usd_tensors()

        # Cache static tensors
        self._to_y_fwd = torch.tensor(
            [0.7071068, -0.7071068, 0.0, 0.0], device=self._device
        )
        self._identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self._device)

        # Cache light data as tensors (stack numpy arrays first to avoid slow path)
        if self._lights:
            light_pos_np = np.array(
                [light.position for light in self._lights], dtype=np.float32
            )
            light_dir_np = np.array(
                [light.direction for light in self._lights], dtype=np.float32
            )
            self._light_positions_cached = torch.from_numpy(light_pos_np).to(
                self._device
            )
            self._light_directions_cached = torch.from_numpy(light_dir_np).to(
                self._device
            )
        else:
            self._light_positions_cached = torch.tensor(
                [[0.0, 10.0, 0.0]], device=self._device, dtype=torch.float32
            )
            self._light_directions_cached = torch.tensor(
                [[0.0, -1.0, 0.0]], device=self._device, dtype=torch.float32
            )

        self._geom_lookup_built = True

        # Pre-expand to_y_fwd for camera transform (avoids per-frame expand)
        self._to_y_fwd_expanded = (
            self._to_y_fwd.unsqueeze(0).expand(self._num_envs, -1).contiguous()
        )

        # Build batched index tensors for fast state updates
        self._build_batched_index_tensors()

    def _precompute_camera_tensors(self, num_cams: int):
        """Pre-compute camera conversion tensors for given num_cams (called once)."""
        if (
            not hasattr(self, "_to_y_fwd_multicam")
            or self._to_y_fwd_multicam.shape[1] != num_cams
        ):
            # (num_envs, num_cams, 4) - pre-expanded for vectorized quat_mul
            self._to_y_fwd_multicam = (
                self._to_y_fwd.unsqueeze(0)
                .unsqueeze(0)
                .expand(self._num_envs, num_cams, -1)
                .reshape(-1, 4)
                .contiguous()
            )

    def _build_batched_index_tensors(self):
        """Build fully batched index tensors for zero-loop state updates.

        Creates pre-computed index tensors that allow gathering ALL articulation
        body poses in a single operation, completely eliminating Python loops.
        """
        # For fully batched articulation updates, we need:
        # 1. A way to gather body_pos/quat from potentially different articulations
        # 2. Output geom indices to scatter to
        #
        # Strategy: Group by unique articulation, build gather indices for each,
        # then combine into single tensors per articulation (can't fully merge
        # across different articulations since they have different data tensors)

        self._articulation_batched = []

        # Group by articulation reference
        artic_to_data: dict[int, tuple[object, list[int], list[int]]] = {}
        for geom_idx, artic_ref, body_idx in zip(
            self._articulation_geom_indices,
            self._articulation_asset_refs,
            self._articulation_body_indices,
        ):
            key = id(artic_ref)
            if key not in artic_to_data:
                artic_to_data[key] = (artic_ref, [], [])
            artic_to_data[key][1].append(geom_idx)
            artic_to_data[key][2].append(body_idx)

        # Build tensor indices for each articulation
        for key, (artic_ref, geom_indices, body_indices) in artic_to_data.items():
            self._articulation_batched.append(
                (
                    artic_ref,
                    torch.tensor(geom_indices, device=self._device, dtype=torch.long),
                    torch.tensor(body_indices, device=self._device, dtype=torch.long),
                )
            )

        # Convert rigid object indices to tensor for batched scatter
        if self._rigid_geom_indices:
            self._rigid_geom_indices_tensor = torch.tensor(
                self._rigid_geom_indices, device=self._device, dtype=torch.long
            )
            # Also store rigid refs in same order
            self._rigid_refs_ordered = self._rigid_asset_refs
            # Build (num_rigid_geoms, 3) scale tensor for per-frame scatter.
            # Broadcast to (1, num_rigid_geoms, 3) so it can be written to
            # geom_scales_out[:, geom_indices, :] in one assignment.
            if self._rigid_geom_scale_list:
                scales_np = np.stack(self._rigid_geom_scale_list, axis=0)  # (N, 3)
            else:
                scales_np = np.ones((len(self._rigid_geom_indices), 3), dtype=np.float32)
            self._rigid_geom_scales_tensor = torch.from_numpy(scales_np).to(
                self._device
            ).unsqueeze(0)  # (1, N, 3) - broadcast over envs
        else:
            self._rigid_geom_indices_tensor = None
            self._rigid_refs_ordered = []
            self._rigid_geom_scales_tensor = None

        # Convert marker indices to tensor
        if self._marker_geom_indices:
            self._marker_geom_indices_tensor = torch.tensor(
                self._marker_geom_indices, device=self._device, dtype=torch.long
            )
        else:
            self._marker_geom_indices_tensor = None

    def _build_static_usd_tensors(self) -> None:
        """Pre-compute geom index tensor and pose tensors for static_usd mappings.

        Positions are stored env-local (world_pos - env_origin) and written
        directly to geom_positions_out at scatter time, bypassing the existing
        env_origins subtraction path used by articulations/rigid_objects.
        """
        # Collect (geom_idx, env_idx, pos_local, quat) tuples from _static_geom_local_poses.
        # In shared_geometry mode only env_0 poses exist; they are broadcast to all envs.
        # In per-env mode each env contributes its own pose list aligned with static_usd mappings.

        # Enumerate static_usd geom indices from _geom_mappings in order.
        static_geom_indices: list[int] = []
        for geom_idx, mapping in enumerate(self._geom_mappings):
            if mapping.asset_type == "static_usd":
                static_geom_indices.append(geom_idx)

        if not static_geom_indices:
            self._static_usd_geom_indices = None
            self._static_usd_positions = None
            self._static_usd_quats = None
            return

        num_static = len(static_geom_indices)

        # Build (num_envs, num_static, 3/4) tensors.
        # Use env_0 poses for all envs in shared mode (kitchen is identical across envs).
        positions_np = np.zeros((self._num_envs, num_static, 3), dtype=np.float32)
        quats_np = np.zeros((self._num_envs, num_static, 4), dtype=np.float32)
        quats_np[:, :, 0] = 1.0  # identity w component

        for env_idx in range(self._num_envs):
            # Fall back to env 0 poses if this env has no recorded poses (shared mode).
            poses = self._static_geom_local_poses.get(
                env_idx, self._static_geom_local_poses.get(0, [])
            )
            if len(poses) != num_static:
                print(
                    f"[MadronaAdapter] WARNING: env {env_idx} has {len(poses)} static poses "
                    f"but {num_static} static geom indices; poses may be misaligned."
                )
            for slot, (pos_local, quat) in enumerate(poses[:num_static]):
                positions_np[env_idx, slot] = pos_local
                quats_np[env_idx, slot] = quat

        self._static_usd_geom_indices = torch.tensor(
            static_geom_indices, device=self._device, dtype=torch.long
        )
        self._static_usd_positions = torch.from_numpy(positions_np).to(self._device)
        self._static_usd_quats = torch.from_numpy(quats_np).to(self._device)

    def extract_geometry(self, env_idx: int = 0) -> MadronaGeometry:
        """One-time extraction of scene geometry from USD.

        Args:
            env_idx: Which environment to extract from. Default 0.
                     Only used if shared_geometry=True.

        Must be called after scene is created but before simulation starts.
        """
        if self._shared_geometry:
            return self._extract_geometry_from_env(env_idx)
        else:
            return self._extract_all_env_geometries()

    def _extract_geometry_from_env(self, env_idx: int = 0) -> MadronaGeometry:
        """Extract geometry from a single environment."""
        geometries: list[SingleGeometry] = []

        for name, articulation in self._env.scene.articulations.items():
            body_geometries = self._extract_articulation_geometry(
                name, articulation, env_idx=env_idx
            )
            geometries.extend(body_geometries)

        for name, rigid_obj in self._scene_rigid_objects().items():
            obj_geom = self._extract_rigid_object_geometry(
                name, rigid_obj, env_idx=env_idx
            )
            if obj_geom is not None:
                geometries.append(obj_geom)

        if self._ingest_stage_meshes:
            static_geoms = self._ingest_usd_stage_meshes(env_idx)
            geometries.extend(static_geoms)

        marker_geometries = self._extract_marker_geometries()
        if not marker_geometries:
            marker_geometries = self._extract_command_marker_fallback_geometries()
        geometries.extend(marker_geometries)

        self._extract_lights()

        self._geometry = combine_geometries(geometries)
        return self._geometry

    def _extract_all_env_geometries(self) -> MadronaGeometry:
        """Extract geometry from all environments (for varied objects per env)."""
        self._per_env_geometries = []
        self._per_env_geom_mappings = []

        for env_idx in range(self._num_envs):
            # Reset mappings for each env
            old_mappings = self._geom_mappings
            self._geom_mappings = []

            geometries: list[SingleGeometry] = []

            for name, articulation in self._env.scene.articulations.items():
                body_geometries = self._extract_articulation_geometry(
                    name, articulation, env_idx=env_idx
                )
                geometries.extend(body_geometries)

            for name, rigid_obj in self._scene_rigid_objects().items():
                obj_geom = self._extract_rigid_object_geometry(
                    name, rigid_obj, env_idx=env_idx
                )
                if obj_geom is not None:
                    geometries.append(obj_geom)

            if self._ingest_stage_meshes:
                static_geoms = self._ingest_usd_stage_meshes(env_idx)
                geometries.extend(static_geoms)

            # Markers are shared (not per-env)
            if env_idx == 0:
                marker_geometries = self._extract_marker_geometries()
                if not marker_geometries:
                    marker_geometries = (
                        self._extract_command_marker_fallback_geometries()
                    )
                geometries.extend(marker_geometries)

            env_geometry = combine_geometries(geometries)
            self._per_env_geometries.append(env_geometry)
            self._per_env_geom_mappings.append(self._geom_mappings.copy())

        # Extract lights once
        self._extract_lights()

        # Use first env's geometry as the main reference
        self._geometry = self._per_env_geometries[0]
        self._geom_mappings = self._per_env_geom_mappings[0]

        return self._geometry

    def get_geometry_for_env(self, env_idx: int) -> MadronaGeometry:
        """Get geometry for a specific environment (only in non-shared mode)."""
        if self._shared_geometry:
            return self._geometry
        if self._per_env_geometries is None:
            raise RuntimeError("Call extract_geometry() first")
        return self._per_env_geometries[env_idx]

    def _extract_marker_geometries(self) -> list[SingleGeometry]:
        """Extract geometry from VisualizationMarkers (PointInstancers)."""
        from pxr import UsdGeom

        geometries = []
        stage = self._get_stage()
        if stage is None:
            return geometries

        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.PointInstancer):
                marker_geoms = self._extract_point_instancer_geometry(prim)
                geometries.extend(marker_geoms)

        return geometries

    def _extract_point_instancer_geometry(self, prim) -> list[SingleGeometry]:
        """Extract geometry from a PointInstancer prim."""
        from pxr import UsdGeom

        geometries = []
        instancer = UsdGeom.PointInstancer(prim)

        proto_indices = instancer.GetProtoIndicesAttr().Get()
        if not proto_indices:
            return geometries

        # Get scales and positions from the instancer
        scales_attr = instancer.GetScalesAttr()
        positions_attr = instancer.GetPositionsAttr()
        orientations_attr = instancer.GetOrientationsAttr()

        scales = scales_attr.Get() if scales_attr.HasValue() else None
        positions = positions_attr.Get() if positions_attr.HasValue() else None
        orientations = orientations_attr.Get() if orientations_attr.HasValue() else None

        prototypes_rel = instancer.GetPrototypesRel()
        prototype_paths = prototypes_rel.GetTargets()

        for proto_idx, proto_path in enumerate(prototype_paths):
            stage = self._get_stage()
            proto_prim = stage.GetPrimAtPath(proto_path)
            if proto_prim.IsValid():

                # Get world scale for the prototype
                world_scale = self._get_world_scale(proto_prim)

                geom = self._extract_prim_geometry(
                    proto_prim, apply_world_scale=world_scale
                )
                if geom is not None:
                    # Make marker semi-transparent green so it's clearly visible as goal
                    geom.color = np.array([0.0, 0.8, 0.0, 0.7], dtype=np.float32)

                    self._geom_mappings.append(
                        GeomMapping(
                            asset_type="marker",
                            asset_name=str(prim.GetPath()),
                            body_index=0,
                            marker_instance_index=len(geometries),
                        )
                    )
                    geometries.append(geom)
                    self._marker_prim_paths.append(str(prim.GetPath()))

        return geometries

    def _extract_command_marker_fallback_geometries(self) -> list[SingleGeometry]:
        """Use object geometry for the command goal marker if no marker prim exists yet.

        In Newton standalone mode the command debug visualizer may not have
        authored its PointInstancer by the time Madrona lazily extracts USD
        geometry during the first observation. The goal marker uses the same
        DexCube asset as the manipulated object, so reusing that extracted
        geometry gives Madrona a stable reference-cube geom slot.
        """
        object_geometries = self._asset_geometry_templates.get("object")
        if not object_geometries:
            return []

        marker_name = "/Visuals/Command/goal_marker"
        try:
            command = self._env.command_manager.get_term("object_pose")
            marker_cfg = getattr(command.cfg, "goal_pose_visualizer_cfg", None)
            marker_name = getattr(marker_cfg, "prim_path", marker_name)
        except Exception:
            pass

        geometries: list[SingleGeometry] = []
        for template in object_geometries:
            geom = self._copy_geometry(template)
            geom.color = np.array([0.0, 0.8, 0.0, 0.7], dtype=np.float32)
            self._geom_mappings.append(
                GeomMapping(
                    asset_type="marker",
                    asset_name=marker_name,
                    body_index=0,
                    marker_instance_index=len(geometries),
                )
            )
            geometries.append(geom)

        print(
            f"[MadronaAdapter] Using object geometry as fallback for goal marker '{marker_name}'."
        )
        return geometries

    def _extract_lights(self):
        """Extract light information from the USD scene."""
        from pxr import UsdGeom, UsdLux

        stage = self._get_stage()
        if stage is None:
            return

        for prim in stage.Traverse():
            light_info = None

            if prim.IsA(UsdLux.DistantLight):
                light = UsdLux.DistantLight(prim)
                # Get light direction from transform
                xform = UsdGeom.Xformable(prim)
                world_xform = xform.ComputeLocalToWorldTransform(0)
                rot_matrix = world_xform.ExtractRotationMatrix()

                # USD DistantLight points along -Z in local space
                # The Z column of rotation matrix gives the local Z axis in world space
                local_z = np.array(
                    [
                        rot_matrix.GetColumn(2)[0],
                        rot_matrix.GetColumn(2)[1],
                        rot_matrix.GetColumn(2)[2],
                    ]
                )
                # Light direction = negative Z (where light rays go)
                direction = -local_z

                # Normalize
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm

                # If direction is [0,0,1] (pointing up/wrong), use a sensible default
                # This happens when the distant light has no rotation or points straight down
                if abs(direction[2]) > 0.99 and direction[2] > 0:
                    # Light pointing up - flip it and add angle
                    direction = np.array([0.3, -0.7, -0.6])
                    direction = direction / np.linalg.norm(direction)
                    print(
                        f"[Lights] DistantLight direction corrected (was pointing up)"
                    )
                elif abs(direction[2]) > 0.99 and direction[2] < 0:
                    # Light pointing straight down - add some angle for better shadows
                    direction = np.array([0.2, -0.3, -0.9])
                    direction = direction / np.linalg.norm(direction)

                light_info = LightInfo(
                    light_type=0,  # Directional
                    position=np.array([0.0, 0.0, 0.0]),
                    direction=direction,
                    color=self._get_light_color(light),
                    intensity=light.GetIntensityAttr().Get() or 1.0,
                    casts_shadow=True,
                )
                if not self._lights_logged:
                    print(
                        f"[Lights] DistantLight: dir={direction}, intensity={light_info.intensity}"
                    )

            elif prim.IsA(UsdLux.SphereLight):
                light = UsdLux.SphereLight(prim)
                xform = UsdGeom.Xformable(prim)
                pos = xform.ComputeLocalToWorldTransform(0).ExtractTranslation()
                light_info = LightInfo(
                    light_type=1,  # Point light
                    position=np.array([pos[0], pos[1], pos[2]]),
                    direction=np.array([0.0, -1.0, 0.0]),  # Not used for point lights
                    color=self._get_light_color(light),
                    intensity=light.GetIntensityAttr().Get() or 1.0,
                    casts_shadow=True,
                )
                if not self._lights_logged:
                    print(
                        f"[Lights] SphereLight: pos={light_info.position}, intensity={light_info.intensity}"
                    )

            elif prim.IsA(UsdLux.DomeLight):
                # Madrona has no HDRI/environment-map support, but the renderer can
                # consume an arbitrary number of directional lights. Approximate the
                # dome by sampling N evenly-distributed directions over the upper
                # hemisphere with intensity = dome_intensity / N. This produces a
                # decent ambient fill that matches the dome's solid color (the
                # texture, if any, is still ignored). For the salad-dressing eval
                # scene with intensity=950 this turns a ~black render into
                # reasonable kitchen lighting.
                light = UsdLux.DomeLight(prim)
                dome_color = self._get_light_color(light)
                dome_intensity = light.GetIntensityAttr().Get() or 1.0
                dome_dirs = [
                    (0.0, 0.0, -1.0),     # straight down (key)
                    (0.7, 0.0, -0.7),     # +X 45 deg
                    (-0.7, 0.0, -0.7),    # -X 45 deg
                    (0.0, 0.7, -0.7),     # +Y 45 deg
                    (0.0, -0.7, -0.7),    # -Y 45 deg
                    (0.0, 0.0, 1.0),      # straight up (sky-side bounce)
                ]
                per_dir_intensity = dome_intensity / len(dome_dirs)
                for dx, dy, dz in dome_dirs:
                    d = np.array([dx, dy, dz], dtype=np.float32)
                    d = d / np.linalg.norm(d)
                    self._lights.append(
                        LightInfo(
                            light_type=0,
                            position=np.array([0.0, 0.0, 0.0]),
                            direction=d,
                            color=dome_color,
                            intensity=per_dir_intensity,
                            casts_shadow=False,
                        )
                    )
                if not self._lights_logged:
                    print(
                        f"[Lights] DomeLight approximated as {len(dome_dirs)} directional lights "
                        f"(intensity={dome_intensity:.1f}, per-dir={per_dir_intensity:.1f}, color={dome_color})"
                    )
                continue

            if light_info is not None:
                self._lights.append(light_info)

        if not self._lights_logged:
            print(f"[Lights] Extracted {len(self._lights)} lights from scene")
        self._lights_logged = True

        # Add default light if none found
        if len(self._lights) == 0:
            # Default: overhead directional light angled toward scene
            default_dir = np.array([0.3, -0.8, -0.5])
            default_dir = default_dir / np.linalg.norm(default_dir)
            self._lights.append(
                LightInfo(
                    light_type=0,
                    position=np.array([0.0, 10.0, 0.0]),
                    direction=default_dir,
                    color=np.array([1.0, 1.0, 1.0]),
                    intensity=1.0,
                    casts_shadow=True,
                )
            )
            print(f"[Lights] Added default directional light: dir={default_dir}")

    def _get_light_color(self, light) -> np.ndarray:
        """Extract color from a USD light."""
        color_attr = light.GetColorAttr()
        if color_attr.HasValue():
            c = color_attr.Get()
            return np.array([c[0], c[1], c[2]])
        return np.array([1.0, 1.0, 1.0])

    def _get_prim_path_from_physx_view(self, asset, env_idx: int = 0) -> str | None:
        """Get the actual prim path from the asset's physx view.

        Args:
            asset: Isaac Lab articulation or rigid object
            env_idx: Which environment's prim path to get

        This is more reliable than trying to resolve cfg.prim_path templates.
        """
        try:
            # Both articulations and rigid objects have root_physx_view.prim_paths
            prim_paths = asset.root_physx_view.prim_paths
            if prim_paths and len(prim_paths) > env_idx:
                return prim_paths[env_idx]
        except (AttributeError, IndexError):
            pass
        return None

    def _resolve_prim_path(self, prim_path: str, env_idx: int = 0) -> str:
        """Resolve Isaac Lab prim path template to actual path for given env.

        Isaac Lab uses patterns like:
        - {ENV_REGEX_NS}/Object -> /World/envs/env_N/Object
        - /World/envs/env_.*/Object -> /World/envs/env_N/Object
        """
        # Replace {ENV_REGEX_NS} first (before it might be expanded)
        if "{ENV_REGEX_NS}" in prim_path:
            prim_path = prim_path.replace(
                "{ENV_REGEX_NS}", f"/World/envs/env_{env_idx}"
            )

        # Handle already-expanded regex patterns: env_.* -> env_N
        if ".*" in prim_path:
            prim_path = prim_path.replace(".*", str(env_idx))

        return prim_path.rstrip("/")

    def _extract_articulation_geometry(
        self, name: str, articulation, env_idx: int = 0
    ) -> list[SingleGeometry]:
        """Extract geometry for each body in an articulation."""

        geometries = []

        # Try to get actual prim path from physx view first
        prim_path = self._get_prim_path_from_physx_view(articulation, env_idx)
        if prim_path is None:
            # Fall back to resolving from cfg
            prim_path = self._resolve_prim_path(articulation.cfg.prim_path, env_idx)

        stage = self._get_stage()
        if stage is None:
            return geometries

        root_prim = stage.GetPrimAtPath(prim_path)
        if not root_prim.IsValid():
            print(f"Warning: Could not find articulation prim at {prim_path}")
            return geometries

        # Newton represents the manipulated cube as an Articulation with no
        # actuators. Apply the same DexCube scale compensation that rigid
        # objects get, without touching articulated robot meshes.
        actuators = getattr(articulation.cfg, "actuators", None)
        apply_world_scale = (
            self._get_world_scale(root_prim) if not actuators else None
        )

        # Get Isaac Lab's body names to map USD body names to PhysX tensor indices
        isaaclab_body_names = articulation.body_names

        # Build a map from body name to PhysX tensor index
        body_name_to_physx_idx = {
            body_name: idx for idx, body_name in enumerate(isaaclab_body_names)
        }

        body_entries = [(prim, None) for prim in self._find_body_prims(root_prim)]
        if not body_entries:
            body_entries = self._find_body_prims_by_name(
                root_prim, body_name_to_physx_idx
            )

        self._articulation_body_counts[name] = len(body_entries)

        for body_prim, known_body_idx in body_entries:
            geom = self._extract_prim_geometry(
                body_prim, apply_world_scale=apply_world_scale
            )
            if geom is not None:
                # Get body name from USD prim path (last component)
                usd_body_name = body_prim.GetName()

                # Find matching PhysX body index
                if known_body_idx is not None:
                    physx_body_idx = known_body_idx
                elif usd_body_name in body_name_to_physx_idx:
                    physx_body_idx = body_name_to_physx_idx[usd_body_name]
                else:
                    # Try to find partial match
                    physx_body_idx = None
                    for il_name, idx in body_name_to_physx_idx.items():
                        if usd_body_name in il_name or il_name in usd_body_name:
                            physx_body_idx = idx
                            break
                    if physx_body_idx is None:
                        print(
                            f"[WARN] Could not match USD body '{usd_body_name}' to any PhysX body, using fallback idx 0"
                        )
                        physx_body_idx = 0

                geometries.append(geom)
                self._geom_mappings.append(
                    GeomMapping(
                        asset_type="articulation",
                        asset_name=name,
                        body_index=physx_body_idx,
                    )
                )

        if not geometries:
            geom = self._extract_prim_geometry(
                root_prim, apply_world_scale=apply_world_scale
            )
            if geom is not None:
                print(
                    f"[MadronaAdapter] Falling back to root geometry for articulation '{name}' at {prim_path}."
                )
                geometries.append(geom)
                self._geom_mappings.append(
                    GeomMapping(
                        asset_type="articulation",
                        asset_name=name,
                        body_index=0,
                    )
                )

        if geometries:
            self._asset_geometry_templates[name] = [
                self._copy_geometry(geom) for geom in geometries
            ]

        return geometries

    def _extract_rigid_object_geometry(
        self, name: str, rigid_obj, env_idx: int = 0
    ) -> SingleGeometry | None:
        """Extract geometry for a rigid object."""
        # Try to get actual prim path from physx view first
        prim_path = self._get_prim_path_from_physx_view(rigid_obj, env_idx)
        if prim_path is None:
            # Fall back to resolving from cfg
            prim_path = self._resolve_prim_path(rigid_obj.cfg.prim_path, env_idx)

        stage = self._get_stage()
        if stage is None:
            return None

        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Could not find rigid object prim at {prim_path}")
            return None

        # Get the world-space scale from the full transform hierarchy.
        # For instanceable USD assets the scale may live on the instance Xform rather
        # than the prototype, so _get_world_scale may return 1.0 when the actual runtime
        # scale is non-unit. We also check cfg.spawn.scale as a fallback.
        world_scale = self._get_world_scale(prim)
        cfg_scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        try:
            spawn = rigid_obj.cfg.spawn
            if hasattr(spawn, "scale") and spawn.scale is not None:
                s = spawn.scale
                if isinstance(s, (int, float)):
                    cfg_scale = np.array([s, s, s], dtype=np.float32)
                else:
                    cfg_scale = np.array(s, dtype=np.float32)
        except Exception:
            pass
        # Prefer cfg_scale when it differs from 1.0 (more reliable for instanceable assets)
        if not np.allclose(cfg_scale, [1.0, 1.0, 1.0], atol=0.001):
            effective_scale = cfg_scale
        else:
            effective_scale = world_scale

        geom = self._extract_prim_geometry(prim, apply_world_scale=effective_scale)
        if geom is not None:

            self._geom_mappings.append(
                GeomMapping(
                    asset_type="rigid_object",
                    asset_name=name,
                    body_index=0,
                )
            )
            self._rigid_asset_scales[name] = effective_scale
            self._asset_geometry_templates[name] = [self._copy_geometry(geom)]

        return geom

    def _ingest_usd_stage_meshes(self, env_idx: int) -> list[SingleGeometry]:
        """Ingest static USD meshes not covered by articulations or rigid objects.

        Walks the env subtree via stage_walker, skipping articulation subtrees and
        already-registered asset paths. Emits GeomMapping(asset_type="static_usd")
        entries and stores env-local poses in self._static_geom_local_poses.

        Returns the list of SingleGeometry objects so callers can extend their
        combined geometry list.
        """
        from pxr import Sdf

        from madrona_mjx_isaaclab.stage_walker import walk_static_meshes

        stage = self._get_stage()
        if stage is None:
            return []

        env_root_path = self._resolve_prim_path("/World/envs/env_.*", env_idx)

        env_origins = self._env.scene.env_origins  # (num_envs, 3)
        env_origin_np = env_origins[env_idx].cpu().numpy()

        # Build covered_paths from articulations and rigid objects already ingested.
        covered_paths: set[Sdf.Path] = set()

        for name, articulation in self._env.scene.articulations.items():
            prim_path = self._get_prim_path_from_physx_view(articulation, env_idx)
            if prim_path is None:
                prim_path = self._resolve_prim_path(articulation.cfg.prim_path, env_idx)
            covered_paths.add(Sdf.Path(prim_path))

        for name, rigid_obj in self._scene_rigid_objects().items():
            prim_path = self._get_prim_path_from_physx_view(rigid_obj, env_idx)
            if prim_path is None:
                prim_path = self._resolve_prim_path(rigid_obj.cfg.prim_path, env_idx)
            covered_paths.add(Sdf.Path(prim_path))

        try:
            records = walk_static_meshes(stage, env_root_path, covered_paths)
        except RuntimeError as exc:
            print(f"[MadronaAdapter] ERROR in walk_static_meshes for env {env_idx}: {exc}")
            raise

        if not records:
            return []

        poses_this_env: list[tuple[np.ndarray, np.ndarray]] = []
        geometries: list[SingleGeometry] = []

        for record in records:
            geom = self._extract_prim_geometry(
                record.prim,
                apply_world_scale=record.scale_world.astype(np.float32),
            )
            if geom is None:
                continue

            pos_local = (record.pos_world - env_origin_np).astype(np.float32)
            quat = record.quat_world_wxyz.astype(np.float32)

            poses_this_env.append((pos_local, quat))
            geometries.append(geom)
            self._geom_mappings.append(
                GeomMapping(
                    asset_type="static_usd",
                    asset_name=str(record.prim.GetPath()),
                    body_index=0,
                )
            )

        self._static_geom_local_poses[env_idx] = poses_this_env
        print(
            f"[MadronaAdapter] env {env_idx}: ingested {len(poses_this_env)} static USD mesh(es)"
        )
        return geometries

    def _get_world_scale(self, prim) -> np.ndarray:
        """Get accumulated world-space scale from prim hierarchy."""
        from pxr import UsdGeom

        scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # Walk up the hierarchy and accumulate scales
        current = prim
        while current.IsValid():
            xformable = UsdGeom.Xformable(current)
            if xformable:
                local_xform = xformable.GetLocalTransformation()
                # Extract scale from transform matrix
                local_scale = np.array(
                    [
                        abs(local_xform.GetRow(0).GetLength()),
                        abs(local_xform.GetRow(1).GetLength()),
                        abs(local_xform.GetRow(2).GetLength()),
                    ],
                    dtype=np.float32,
                )
                scale = scale * local_scale
            current = current.GetParent()

        return scale

    def _find_body_prims(self, root_prim) -> list:
        """Find all body prims in an articulation hierarchy."""
        from pxr import UsdPhysics

        body_prims = []

        def traverse(prim):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_prims.append(prim)
            for child in prim.GetChildren():
                traverse(child)

        traverse(root_prim)
        return body_prims

    def _find_body_prims_by_name(
        self, root_prim, body_name_to_idx: dict[str, int]
    ) -> list[tuple[object, int]]:
        """Find articulation body prims by Isaac Lab body names.

        IsaacLabNewton does not always leave UsdPhysics.RigidBodyAPI on the USD
        prims that hold visuals, so the PhysX-oriented schema search can return
        nothing. Body/link names are still present in the USD hierarchy.
        """
        matches: list[tuple[object, int]] = []
        matched_indices: set[int] = set()

        def match_body_idx(prim_name: str) -> int | None:
            if prim_name in body_name_to_idx:
                return body_name_to_idx[prim_name]
            for body_name, body_idx in body_name_to_idx.items():
                if prim_name in body_name or body_name in prim_name:
                    return body_idx
            return None

        def traverse(prim):
            body_idx = match_body_idx(prim.GetName())
            if body_idx is not None and body_idx not in matched_indices:
                matches.append((prim, body_idx))
                matched_indices.add(body_idx)
                return
            for child in prim.GetChildren():
                traverse(child)

        traverse(root_prim)
        if matches:
            return matches

        geom = self._extract_prim_geometry(root_prim)
        if geom is not None:
            return [(root_prim, 0)]

        return []

    def _extract_prim_geometry(
        self, prim, depth: int = 0, apply_world_scale: np.ndarray | None = None
    ) -> SingleGeometry | None:
        """Extract geometry from a USD prim using appropriate adapter.

        Searches through the prim hierarchy to find visual geometry.
        Isaac Sim assets typically have geometry under 'visuals' child.

        Args:
            prim: USD prim to extract geometry from
            depth: Current recursion depth
            apply_world_scale: World scale from the object root prim. When provided,
                              we compensate for mesh prim's local scale to get correct world size.
        """
        from pxr import UsdGeom

        prim_type = prim.GetTypeName()
        geom = None

        # Check if this prim itself is geometry
        adapter = GeometryAdapterFactory.create(
            prim_type, max_texture_size=self._max_texture_size
        )
        if adapter:
            geom = adapter.extract(prim)

        # Check if prim is a Mesh, Cube, or Sphere
        if geom is None and prim.IsA(UsdGeom.Mesh):
            adapter = GeometryAdapterFactory.create(
                "Mesh", max_texture_size=self._max_texture_size
            )
            if adapter:
                geom = adapter.extract(prim)
        elif geom is None and prim.IsA(UsdGeom.Cube):
            adapter = GeometryAdapterFactory.create("Cube")
            if adapter:
                geom = adapter.extract(prim)
        elif geom is None and prim.IsA(UsdGeom.Sphere):
            adapter = GeometryAdapterFactory.create("Sphere")
            if adapter:
                geom = adapter.extract(prim)

        # If we found geometry, apply scale correction
        if geom is not None:
            if apply_world_scale is not None:
                # The adapter applied the mesh prim's local transform (including scale).
                # For USD files like DexCube where mesh has scale 0.03 baked in,
                # we need to compensate by dividing by local_scale to get correct world size.
                #
                # Example: DexCube mesh vertices are [-0.03, 0.03] (6cm cube in model space)
                # Mesh prim has local_scale=0.03, so after adapter: [-0.0009, 0.0009]
                # We apply parent_scale = world_scale / local_scale = 1/0.03 = 33.33
                # Result: [-0.03, 0.03] (6cm cube) - correct!
                xformable = UsdGeom.Xformable(prim)
                local_xform = xformable.GetLocalTransformation()
                local_scale = np.array(
                    [
                        abs(local_xform.GetRow(0).GetLength()),
                        abs(local_xform.GetRow(1).GetLength()),
                        abs(local_xform.GetRow(2).GetLength()),
                    ],
                    dtype=np.float32,
                )

                # parent_scale = object_world_scale / mesh_local_scale
                parent_scale = apply_world_scale / np.maximum(local_scale, 0.0001)
                if not np.allclose(parent_scale, [1, 1, 1], atol=0.001):
                    geom.vertices = geom.vertices * parent_scale
            return geom

        # Look for a 'visuals'/'Visuals' child first (Isaac Sim/Robocasa convention).
        # Robocasa rigid_objects also have a 'Collisions' sibling with collision-only meshes
        # that have NO bound material -- if we descend there first, color extraction falls
        # back to the default gray and the object renders flat-white.
        for visuals_name in ("visuals", "Visuals"):
            visuals_child = prim.GetChild(visuals_name)
            if visuals_child.IsValid():
                result = self._extract_prim_geometry(
                    visuals_child, depth + 1, apply_world_scale
                )
                if result is not None:
                    return result

        # Check for instanceable prims - mesh may be in a prototype
        # Instanceable USD assets store their mesh in stage.GetPrototypes()
        if prim.IsInstance():
            prototype = prim.GetPrototype()
            if prototype and prototype.IsValid():
                result = self._extract_prim_geometry(
                    prototype, depth + 1, apply_world_scale
                )
                if result is not None:
                    return result

        # Also check stage prototypes (for instanceable meshes referenced via composition)
        stage = self._get_stage()
        if stage and depth == 0:  # Only at top level to avoid infinite recursion
            prototypes = stage.GetPrototypes()
            if prototypes:
                for proto in prototypes:
                    # Try to find a matching prototype by name
                    prim_name = prim.GetName().lower()
                    proto_name = proto.GetPath().name.lower()
                    if (
                        prim_name in proto_name
                        or proto_name in prim_name
                        or "cube" in proto_name
                    ):
                        result = self._extract_prim_geometry(
                            proto, depth + 1, apply_world_scale
                        )
                        if result is not None:
                            return result

        # Recursively search all children (limit depth to avoid infinite loops).
        # Skip purely-collision subtrees so we don't return a Mesh with no material binding.
        if depth < 10:
            for child in prim.GetChildren():
                child_name = child.GetName()
                if child_name in ("Collisions", "collisions", "Collision", "collision"):
                    continue
                result = self._extract_prim_geometry(
                    child, depth + 1, apply_world_scale
                )
                if result is not None:
                    return result

        return None

    def _get_stage(self):
        """Get the current USD stage."""
        try:
            from isaaclab.sim.utils.stage import get_current_stage

            return get_current_stage()
        except Exception:
            return None

    def _is_newton_articulation_data(self, data) -> bool:
        """Return true for IsaacLab Newton articulation data objects."""
        return "isaaclab_newton" in type(data).__module__

    def _gather_articulation_body_poses(
        self,
        articulation,
        body_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather body poses as Torch tensors in Madrona's expected format."""
        data = articulation.data

        if self._is_newton_articulation_data(data):
            import warp as wp

            # Newton stores body transforms in one Warp array as [pos, quat_xyzw].
            # Use it directly instead of the high-overhead split position/quaternion properties.
            pose = wp.to_torch(data.body_link_pose_w).index_select(1, body_indices)
            pos = pose[..., :3]
            quat = torch.roll(pose[..., 3:7], shifts=1, dims=-1)
            return pos, quat

        pos = data.body_pos_w.index_select(1, body_indices)
        quat = data.body_quat_w.index_select(1, body_indices)
        return pos, quat

    def build_state_inplace(
        self,
        geom_positions_out: torch.Tensor,
        geom_quaternions_out: torch.Tensor,
        geom_scales_out: torch.Tensor,
        cam_pos_out: torch.Tensor,
        cam_quat_out: torch.Tensor,
        light_positions_out: torch.Tensor,
        light_directions_out: torch.Tensor,
        camera_positions: torch.Tensor,
        camera_quaternions: torch.Tensor,
        marker_positions: dict[str, torch.Tensor] | None = None,
        marker_quaternions: dict[str, torch.Tensor] | None = None,
        camera_convention: str = "world",
    ) -> None:
        """Build per-frame state in-place into pre-allocated buffers.

        Optimized version using pre-built lookup tables to avoid per-frame
        dict lookups and string comparisons. ~15-25% faster than original.

        Args:
            geom_positions_out: Pre-allocated output buffer (num_envs, num_geoms, 3)
            geom_quaternions_out: Pre-allocated output buffer (num_envs, num_geoms, 4)
            geom_scales_out: Pre-allocated output buffer (num_envs, num_geoms, 3)
            cam_pos_out: Pre-allocated output buffer (num_envs, num_cams, 3)
            cam_quat_out: Pre-allocated output buffer (num_envs, num_cams, 4)
            light_positions_out: Pre-allocated output buffer (num_envs, num_lights, 3)
            light_directions_out: Pre-allocated output buffer (num_envs, num_lights, 3)
            camera_positions: Camera positions (num_envs, 3) or (num_envs, num_cams, 3)
            camera_quaternions: Camera quaternions wxyz (num_envs, 4) or (num_envs, num_cams, 4)
            marker_positions: Optional dict mapping marker names to positions
            marker_quaternions: Optional dict mapping marker names to quaternions
            camera_convention: Convention of camera_quaternions ("world", "opengl", "ros")
        """
        # Build lookup tables on first call (lazy init)
        if not self._geom_lookup_built:
            self._build_geom_lookup_tables()

        env_origins = self._env.scene.env_origins  # (num_envs, 3)

        # Reset scales to 1.0 (in-place)
        geom_scales_out.fill_(1.0)

        # Process articulations using batched index_select + JIT scatter
        env_origins_expanded = env_origins.unsqueeze(1)  # (num_envs, 1, 3)
        for articulation, geom_indices, body_indices in self._articulation_batched:
            # Gather all body poses for this articulation in one operation
            pos, quat = self._gather_articulation_body_poses(
                articulation, body_indices
            )
            # JIT scatter kernel
            _jit_scatter_poses(
                geom_positions_out,
                geom_quaternions_out,
                pos,
                quat,
                geom_indices,
                env_origins_expanded,
            )

        # Process rigid objects - still need loop but fewer iterations
        for geom_idx, rigid_obj in zip(
            self._rigid_geom_indices, self._rigid_asset_refs
        ):
            pos = rigid_obj.data.root_pos_w
            quat = rigid_obj.data.root_quat_w
            torch.sub(pos, env_origins, out=geom_positions_out[:, geom_idx, :])
            geom_quaternions_out[:, geom_idx, :].copy_(quat)

        # Write per-frame scales for rigid objects.
        # geom_scales_out was filled with 1.0 above; overwrite only rigid geom slots
        # with their init-time extracted scale (broadcast over envs).
        if (
            self._rigid_geom_indices_tensor is not None
            and self._rigid_geom_scales_tensor is not None
        ):
            geom_scales_out[:, self._rigid_geom_indices_tensor, :] = (
                self._rigid_geom_scales_tensor
            )

        # Process markers using cached indices
        for geom_idx, asset_name in zip(
            self._marker_geom_indices, self._marker_asset_names
        ):
            if marker_positions and asset_name in marker_positions:
                pos = marker_positions[asset_name]
                quat = (
                    marker_quaternions.get(
                        asset_name, self._identity_quat.expand(self._num_envs, -1)
                    )
                    if marker_quaternions
                    else self._identity_quat.expand(self._num_envs, -1)
                )
                torch.sub(pos, env_origins, out=geom_positions_out[:, geom_idx, :])
                geom_quaternions_out[:, geom_idx, :].copy_(quat)
            else:
                geom_positions_out[:, geom_idx, :].zero_()
                geom_quaternions_out[:, geom_idx, 0] = 1.0
                geom_quaternions_out[:, geom_idx, 1:].zero_()

        # Scatter static USD meshes (constant pose, stored env-local — write directly,
        # no env_origins subtraction since positions are already env-local).
        if self._static_usd_geom_indices is not None and self._static_usd_geom_indices.numel():
            geom_positions_out[:, self._static_usd_geom_indices, :] = self._static_usd_positions
            geom_quaternions_out[:, self._static_usd_geom_indices, :] = self._static_usd_quats

        # Handle camera positions (fully vectorized)
        num_cams = cam_pos_out.shape[1]
        if camera_positions.dim() == 3:
            # (num_envs, num_cams, 3) - (num_envs, 1, 3) -> (num_envs, num_cams, 3)
            torch.sub(camera_positions, env_origins.unsqueeze(1), out=cam_pos_out)
        else:
            torch.sub(camera_positions, env_origins, out=cam_pos_out[:, 0, :])

        # Camera orientation conversion (fully vectorized - no loops)
        if camera_quaternions.dim() == 3:
            # Multi-cam: (num_envs, num_cams, 4)
            # Lazy init pre-computed tensor for this num_cams
            self._precompute_camera_tensors(num_cams)

            if camera_convention == "opengl":
                quat_opengl_flat = camera_quaternions.reshape(-1, 4)
            elif camera_convention == "ros":
                quat_opengl_flat = convert_camera_frame_orientation_convention(
                    camera_quaternions.reshape(-1, 4), origin="ros", target="opengl"
                )
            else:
                quat_opengl_flat = convert_camera_frame_orientation_convention(
                    camera_quaternions.reshape(-1, 4), origin="world", target="opengl"
                )
            # JIT camera transform kernel
            _jit_camera_transform(
                cam_pos_out,
                cam_quat_out,
                camera_positions,
                quat_opengl_flat,
                env_origins.unsqueeze(1),
                self._to_y_fwd_multicam,
                self._num_envs,
                num_cams,
            )
        else:
            if camera_convention == "opengl":
                cam_quat_opengl = camera_quaternions
            elif camera_convention == "ros":
                cam_quat_opengl = convert_camera_frame_orientation_convention(
                    camera_quaternions, origin="ros", target="opengl"
                )
            else:
                cam_quat_opengl = convert_camera_frame_orientation_convention(
                    camera_quaternions, origin="world", target="opengl"
                )
            cam_quat_madrona = _jit_quat_mul(cam_quat_opengl, self._to_y_fwd_expanded)
            cam_quat_out[:, 0, :].copy_(cam_quat_madrona)

        # Copy light data using cached tensors (no per-frame tensor creation)
        num_lights = self._light_positions_cached.shape[0]
        light_positions_out[:, :num_lights, :] = self._light_positions_cached
        light_directions_out[:, :num_lights, :] = self._light_directions_cached

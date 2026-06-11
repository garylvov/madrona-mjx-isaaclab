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

"""Madrona-backed tiled camera for fast GPU rasterization.

Drop-in replacement for TiledCamera that uses Madrona's BatchRenderer
instead of RTX for 10-100x faster rendering.

Supports both single-camera and multi-camera modes:
- Single: cfg.offset defines one camera per env
- Multi: cfg.cameras defines N cameras per env
"""

from __future__ import annotations

import os
import time

# CRITICAL: Limit JAX memory BEFORE importing JAX
if "XLA_PYTHON_CLIENT_MEM_FRACTION" not in os.environ:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.05"

# CRITICAL: Constrain JAX to this rank's GPU before any other JAX op.
# jax.devices("gpu") otherwise enumerates every visible CUDA device on first call
# and conflicts with primary contexts already held by sibling ranks (flag=AUTO
# vs torch's BLOCKING_SYNC), corrupting XLA state and segfaulting Madrona's
# BatchRenderer downstream. jax.distributed.initialize(local_device_ids=...) is
# JAX's documented multi-process hook: it pins this process to exactly one
# local device. We deliberately do not touch CUDA_VISIBLE_DEVICES because
# torch's NCCL collectives need all GPUs visible.
if os.environ.get("LOCAL_RANK") is not None and "WORLD_SIZE" in os.environ:
    import jax as _jax_for_init
    if not getattr(_jax_for_init.distributed, "_GIGASTRAP_INIT", False):
        try:
            _master_port = int(os.environ.get("MASTER_PORT", "29500"))
            _jax_for_init.distributed.initialize(
                coordinator_address=f"localhost:{_master_port + 1}",
                num_processes=int(os.environ["WORLD_SIZE"]),
                process_id=int(os.environ["RANK"]),
                local_device_ids=[int(os.environ["LOCAL_RANK"])],
            )
            _jax_for_init.distributed._GIGASTRAP_INIT = True
            print(f"[MadronaTiledCamera] jax.distributed.initialize bound rank "
                  f"{os.environ['RANK']} to local device {os.environ['LOCAL_RANK']}")
        except (TypeError, RuntimeError, OSError) as _e:
            print(f"[MadronaTiledCamera] WARN: jax.distributed.initialize failed: {_e}; "
                  f"falling back to per-process default-device pinning only")

# Profiling control via environment variable
_PROFILE_ENABLED = os.environ.get("MADRONA_PROFILE", "0") == "1"

# Debug image saving control - set to save first rendered image for debugging
_SAVE_FIRST_IMAGE = os.environ.get("MADRONA_SAVE_FIRST_IMAGE", "0") == "1"
if _SAVE_FIRST_IMAGE:
    print(
        "[MadronaTiledCamera] MADRONA_SAVE_FIRST_IMAGE=1: Will save first rendered image to /tmp/madrona_debug/"
    )

_FORCE_COLOR_OVERRIDES = os.environ.get("MADRONA_FORCE_COLOR_OVERRIDES", "0") == "1"
if _FORCE_COLOR_OVERRIDES:
    print(
        "[MadronaTiledCamera] MADRONA_FORCE_COLOR_OVERRIDES=1: Forcing per-geometry debug colors."
    )

import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax.dlpack as jax_dlpack

# Module-level JAX imports (moved from per-frame methods)
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack


class RenderProfiler:
    """Simple profiler for render pipeline stages."""

    def __init__(self):
        self.timings: dict[str, list[float]] = {}
        self.call_count = 0
        self.report_interval = int(os.environ.get("MADRONA_PROFILE_INTERVAL", "10"))

    def record(self, name: str, duration: float):
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

    def tick(self):
        self.call_count += 1
        if self.call_count % self.report_interval == 0:
            self.report()

    def report(self):
        if not self.timings:
            return
        print(
            f"\n[MadronaProfiler] After {self.call_count} frames (last {self.report_interval}):"
        )
        total = 0.0
        # Sort by timing name for consistent ordering
        for name, times in sorted(self.timings.items()):
            avg_ms = sum(times) / len(times) * 1000
            min_ms = min(times) * 1000
            max_ms = max(times) * 1000
            total += avg_ms
            print(f"  {name}: {avg_ms:.3f}ms avg (min={min_ms:.3f}, max={max_ms:.3f})")
        print(f"  TOTAL: {total:.3f}ms per frame")
        # Clear for next interval
        self.timings.clear()


_profiler = RenderProfiler() if _PROFILE_ENABLED else None

from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.sensors.sensor_base import SensorBase

from madrona_mjx_isaaclab.usd_chain import USDArticulationChain

from .adapter import SceneToMadronaAdapter
from .buffer_pool import BufferSpec, PersistentBufferPool
from .geometry import MadronaGeometry

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .camera_cfg import MadronaMultiCamTiledCameraCfg, MadronaTiledCameraCfg

# Global registry for environment references
_ENV_REGISTRY: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def register_env_for_madrona(env: "ManagerBasedRLEnv") -> None:
    """Register an environment for MadronaTiledCamera to access."""
    _ENV_REGISTRY["current"] = env


def _get_registered_env() -> "ManagerBasedRLEnv":
    """Get the registered environment."""
    env = _ENV_REGISTRY.get("current")
    if env is None:
        raise RuntimeError(
            "No environment registered for MadronaTiledCamera. "
            "Call register_env_for_madrona(env) after creating the environment."
        )
    return env


class MadronaTiledCamera(SensorBase):
    """GPU-accelerated tiled camera using Madrona rasterization.

    Drop-in replacement for TiledCamera that uses Madrona's BatchRenderer
    instead of RTX for 10-100x faster rendering at the cost of visual fidelity.

    Supports both single-camera and multi-camera modes:
    - Single camera: Use MadronaTiledCameraCfg with cfg.offset
    - Multi camera: Use MadronaMultiCamTiledCameraCfg with cfg.cameras list

    Multi-camera output keys:
    - "rgb": First camera (num_envs, H, W, 3)
    - "rgb_1": Second camera (num_envs, H, W, 3)
    - etc.

    Supported data types: "rgb", "rgba", "depth"

    Debug image saving:
    - Set MADRONA_SAVE_FIRST_IMAGE=1 to save the first rendered image to /tmp/madrona_debug/
    """

    SUPPORTED_DATA_TYPES = frozenset(["rgb", "rgba", "depth"])

    # Class-level flag to track if first image has been saved (shared across instances)
    _first_image_saved = False

    # Class-level flag to log scene geometry summary only once across all GPU instances
    _geometry_logged = False

    cfg: "MadronaTiledCameraCfg | MadronaMultiCamTiledCameraCfg"

    def __init__(self, cfg: "MadronaTiledCameraCfg | MadronaMultiCamTiledCameraCfg"):
        """Initialize Madrona-backed tiled camera.

        Args:
            cfg: Camera configuration (single or multi-cam).

        Raises:
            ValueError: If unsupported data types are requested.
        """
        unsupported = set(cfg.data_types) - self.SUPPORTED_DATA_TYPES
        if unsupported:
            raise ValueError(
                f"MadronaTiledCamera does not support: {unsupported}. "
                f"Supported types: {self.SUPPORTED_DATA_TYPES}"
            )

        super().__init__(cfg)

        # Detect single vs multi-camera mode from config
        self._num_cams = self._detect_num_cams(cfg)
        self._camera_offsets = self._normalize_offsets(cfg)
        self._camera_spawns = self._normalize_spawns(cfg)

        self._renderer = None
        self._render_token = None
        self._scene_adapter: SceneToMadronaAdapter | None = None
        self._data = CameraData()

        # Camera poses: (num_envs, num_cams, 3) and (num_envs, num_cams, 4).
        # Keep quaternions in named conventions so CameraData remains IsaacLab-compatible.
        self._static_camera_pos: torch.Tensor | None = None
        self._static_camera_quat_world: torch.Tensor | None = None
        self._static_camera_quat_opengl: torch.Tensor | None = None

        # Dynamic camera bookkeeping (populated by _init_camera_poses)
        self._dyn_camera_mask: list[int] = []
        self._dyn_grouped: dict = {}
        self._dyn_offset_pos: torch.Tensor | None = None
        self._dyn_offset_quat_opengl_wxyz: torch.Tensor | None = None

        self._madrona_available = self._check_madrona_available()
        self._madrona_initialized = False
        self._madrona_renderer = None
        self._init_fn = None
        self._render_fn = None
        self._needs_init = True

        self._num_envs = None
        self._device = None
        self._frame = None

        # Persistent buffer pool for zero-copy GPU memory sharing
        self._buffer_pool: PersistentBufferPool | None = None

        # Track if depth is needed (skip conversion if RGB-only)
        self._needs_depth = "depth" in self.cfg.data_types

        # Goal-image render pass state. Lazily populated on first
        # _update_buffers_impl after the articulation is initialized.
        self._render_goal_pass: bool = bool(getattr(self.cfg, "render_goal_pass", False))
        self._fk_chain: USDArticulationChain | None = None
        self._goal_qpos_lag_check_remaining: int = 5 if self._render_goal_pass else 0
        # Resolved lazily once num_cams is known (after _detect_num_cams below)
        self._goal_cam_indices: tuple[int, ...] | None = None

    def _detect_num_cams(self, cfg) -> int:
        """Detect number of cameras from config type."""
        cameras = getattr(cfg, "cameras", None)
        if cameras:
            if len(cameras) < 2:
                raise ValueError("Madrona multicam requires at least two camera configs.")
            return len(cameras)
        return 1

    def _normalize_offsets(self, cfg) -> list[tuple]:
        """Normalize config offsets to list of (pos, rot, convention, body_name, asset_name) tuples."""
        cameras = getattr(cfg, "cameras", None)
        if cameras:
            return [
                (
                    c.offset.pos,
                    c.offset.rot,
                    getattr(c.offset, "convention", "world"),
                    getattr(c.offset, "body_name", None),
                    getattr(c.offset, "asset_name", "robot"),
                )
                for c in cameras
            ]
        return [
            (
                cfg.offset.pos,
                cfg.offset.rot,
                getattr(cfg.offset, "convention", "world"),
                getattr(cfg.offset, "body_name", None),
                getattr(cfg.offset, "asset_name", "robot"),
            )
        ]

    def _normalize_spawns(self, cfg) -> list:
        """Normalize config spawn settings to one pinhole config per camera."""
        cameras = getattr(cfg, "cameras", None)
        if cameras:
            spawns = [c.spawn for c in cameras]
        else:
            spawns = [cfg.spawn]
        if len(spawns) != self._num_cams:
            raise ValueError(f"Expected {self._num_cams} camera spawns, got {len(spawns)}.")
        for cam_idx, spawn in enumerate(spawns):
            self._validate_pinhole_spawn(spawn, cam_idx)
        return spawns

    def _check_madrona_available(self) -> bool:
        """Check if Madrona is installed."""
        try:
            from madrona_mjx import BatchRenderer

            return True
        except ImportError:
            return False

    @property
    def data(self) -> CameraData:
        """Camera data including rendered images."""
        self._update_outdated_buffers()
        return self._data

    @property
    def num_instances(self) -> int:
        """Number of camera instances (one per environment)."""
        if hasattr(self, "_num_envs") and self._num_envs is not None:
            return self._num_envs
        return getattr(super(), "num_instances", 0)

    @property
    def num_cameras(self) -> int:
        """Number of cameras per environment."""
        return self._num_cams

    def reset(self, env_ids: Sequence[int] | None = None, env_mask=None):
        """Reset camera state for specified environments."""
        env = _ENV_REGISTRY.get("current")
        if (
            env is not None
            and self._num_envs is not None
            and self._num_envs != env.num_envs
        ):
            actual_num_envs = env.num_envs
            print(
                f"[MadronaTiledCamera] reset: Resizing buffers {self._num_envs} -> {actual_num_envs}"
            )
            self._num_envs = actual_num_envs
            self._device = env.device
            self._is_outdated = torch.ones(
                self._num_envs, dtype=torch.bool, device=self._device
            )
            self._timestamp = torch.zeros(self._num_envs, device=self._device)
            self._timestamp_last_update = torch.zeros_like(self._timestamp)
            if hasattr(self, "_frame") and self._frame is not None:
                self._frame = torch.zeros(
                    self._num_envs, device=self._device, dtype=torch.long
                )
            self._create_placeholder_buffers()

        super().reset(env_ids)
        if hasattr(self, "_frame") and self._frame is not None:
            if env_ids is None:
                env_ids = slice(None)
            self._frame[env_ids] = 0

    def _initialize_impl(self):
        """Initialize Madrona renderer with scene geometry."""
        super()._initialize_impl()

        if not self._madrona_available:
            raise RuntimeError("Madrona BatchRenderer not found. Install madrona_mjx with 'pixi run -e gsi doit utils:build-madrona-mjx'")

        old_num_envs = self._num_envs
        if hasattr(self, "_view") and self._view is not None:
            self._num_envs = self._view.count

        self._frame = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)

        if old_num_envs != self._num_envs:
            self._is_outdated = torch.ones(
                self._num_envs, dtype=torch.bool, device=self._device
            )
            self._timestamp = torch.zeros(self._num_envs, device=self._device)
            self._timestamp_last_update = torch.zeros_like(self._timestamp)

        self._create_placeholder_buffers()
        self._madrona_initialized = False
        self._scene_adapter = None
        if not MadronaTiledCamera._geometry_logged:
            print(
                f"[MadronaTiledCamera] Initialized with {self._num_envs} envs, {self._num_cams} camera(s)"
            )

    def _output_key(self, data_type: str, cam_idx: int) -> str:
        """Generate output key for given data type and camera index.

        Single cam: "rgb", "depth"
        Multi cam: "rgb" (cam 0), "rgb_1" (cam 1), "rgb_2" (cam 2), ...
        """
        if cam_idx == 0:
            return data_type
        return f"{data_type}_{cam_idx}"

    def _goal_output_key(self, cam_idx: int) -> str:
        """Output key for the goal-pass RGB image."""
        if cam_idx == 0:
            return "goal_rgb"
        return f"goal_rgb_{cam_idx}"

    def _resolve_goal_cam_indices(self) -> tuple[int, ...]:
        """Which cam indices actually emit goal_rgb. Cached after first resolve."""
        if self._goal_cam_indices is not None:
            return self._goal_cam_indices
        cfg_indices = getattr(self.cfg, "goal_camera_indices", None)
        if cfg_indices is None:
            self._goal_cam_indices = tuple(range(self._num_cams))
        else:
            valid = tuple(i for i in cfg_indices if 0 <= i < self._num_cams)
            if not valid:
                print(
                    f"[MadronaTiledCamera] goal_camera_indices={cfg_indices} "
                    f"contains no valid index for num_cams={self._num_cams}; "
                    f"defaulting to (0,)."
                )
                valid = (0,)
            self._goal_cam_indices = valid
        return self._goal_cam_indices

    def _create_placeholder_buffers(self):
        """Create placeholder output buffers for shape queries."""
        height = self.cfg.height
        width = self.cfg.width

        self._data.output = {}
        for data_type in self.cfg.data_types:
            for cam_idx in range(self._num_cams):
                key = self._output_key(data_type, cam_idx)

                if data_type in ("rgb", "rgba"):
                    channels = 4 if data_type == "rgba" else 3
                    self._data.output[key] = torch.zeros(
                        (self._num_envs, height, width, channels),
                        device=self._device,
                        dtype=torch.uint8,
                    )
                elif data_type == "depth":
                    self._data.output[key] = torch.full(
                        (self._num_envs, height, width, 1),
                        float("inf"),
                        device=self._device,
                        dtype=torch.float32,
                    )

        if self._render_goal_pass:
            for cam_idx in self._resolve_goal_cam_indices():
                key = self._goal_output_key(cam_idx)
                self._data.output[key] = torch.zeros(
                    (self._num_envs, height, width, 3),
                    device=self._device,
                    dtype=torch.uint8,
                )

    def _lazy_init_madrona(self):
        """Lazily initialize Madrona when environment is available."""
        if self._madrona_initialized:
            return

        env = self._get_env()
        actual_num_envs = env.num_envs
        actual_device = env.device

        # Goal pass invariant: relies on observation_manager.compute() being the
        # only render trigger per step (lazy_sensor_update=True). If sensors are
        # force-recomputed inside scene.update(dt), joint_pos_target won't yet
        # hold step N's setpoint when this fires.
        if self._render_goal_pass:
            scene_cfg = getattr(env.scene, "cfg", None)
            lazy = getattr(scene_cfg, "lazy_sensor_update", True) if scene_cfg else True
            if not lazy:
                raise RuntimeError(
                    "MadronaTiledCamera.render_goal_pass requires "
                    "InteractiveSceneCfg.lazy_sensor_update=True; goal-pass "
                    "qpos source assumes observation_manager.compute is the "
                    "render trigger. Got lazy_sensor_update=False."
                )

        # Set CUDA device context before any allocations to prevent GPU 0 bloat
        gpu_id = 0
        if "cuda" in str(actual_device):
            try:
                gpu_id = int(str(actual_device).split(":")[-1])
            except (ValueError, IndexError):
                gpu_id = 0
        torch.cuda.set_device(gpu_id)

        # Set JAX default device to prevent JAX preallocation on GPU 0
        # Only do this once per process to avoid conflicts
        import jax

        if not hasattr(MadronaTiledCamera, "_jax_device_configured"):
            try:
                jax_devices = jax.devices("gpu")
                if gpu_id < len(jax_devices):
                    jax.config.update("jax_default_device", jax_devices[gpu_id])
                    print(
                        f"[MadronaTiledCamera] Set JAX default device to {jax_devices[gpu_id]}"
                    )
                MadronaTiledCamera._jax_device_configured = True
            except Exception as e:
                print(
                    f"[MadronaTiledCamera] Warning: Could not set JAX default device: {e}"
                )

        if self._num_envs != actual_num_envs:
            print(
                f"[MadronaTiledCamera] Resizing buffers: {self._num_envs} -> {actual_num_envs} envs"
            )
            self._num_envs = actual_num_envs
            self._device = actual_device
            self._frame = torch.zeros(
                self._num_envs, device=self._device, dtype=torch.long
            )
            self._is_outdated = torch.ones(
                self._num_envs, dtype=torch.bool, device=self._device
            )
            self._timestamp = torch.zeros(self._num_envs, device=self._device)
            self._timestamp_last_update = torch.zeros_like(self._timestamp)
            self._create_placeholder_buffers()

        self._device = actual_device

        self._scene_adapter = SceneToMadronaAdapter(
            env, max_texture_size=getattr(self.cfg, "max_texture_size", None)
        )
        geometry = self._scene_adapter.extract_geometry()
        if self._scene_adapter.num_geoms == 0 or len(geometry.vertices) == 0:
            raise RuntimeError(
                "Madrona scene extraction produced zero geometry. "
                "Check that the active backend exposes USD visual meshes under the scene articulations."
            )

        if not MadronaTiledCamera._geometry_logged:
            num_uvs = len(geometry.uvs) if len(geometry.uvs) > 0 else 0
            num_textures = len(geometry.textures) if geometry.textures else 0
            print(
                f"[MadronaTiledCamera] Scene: {self._scene_adapter.num_geoms} geometries, "
                f"{len(geometry.vertices)} vertices, {len(geometry.indices)} triangles, "
                f"{num_uvs} UVs, {num_textures} textures"
            )
            if geometry.textures:
                n_show = min(5, len(geometry.textures))
                print(f"[MadronaTiledCamera] First {n_show} texture src paths:")
                for i, tex in enumerate(geometry.textures[:n_show]):
                    path_attr = getattr(tex, "src_path", None) or getattr(tex, "path", None)
                    if path_attr:
                        print(f"[MadronaTiledCamera]   tex{i}: {path_attr}")
                    else:
                        print(f"[MadronaTiledCamera]   tex{i}: shape={tex.pixels.shape}")
            if geometry.geom_texture_ids:
                hist: dict[int, int] = {}
                for tid in geometry.geom_texture_ids:
                    hist[tid] = hist.get(tid, 0) + 1
                print(f"[MadronaTiledCamera] texture geom_count_per_tex_id histogram: {hist}")
            MadronaTiledCamera._geometry_logged = True

        self._init_camera_poses()
        self._init_madrona_renderer(geometry)
        self._init_buffer_pool()
        self._create_buffers()

        self._madrona_initialized = True

        # Barrier: wait for all ranks to finish Madrona init before proceeding.
        # The serialized file lock means ranks finish at very different times;
        # without this barrier, early ranks race ahead into the training loop
        # while late ranks are still compiling shaders/CUDA kernels, causing
        # distributed sync deadlocks and heap corruption under VRAM pressure.
        if torch.distributed.is_initialized():
            gpu_id_str = str(self._device).split(":")[-1] if "cuda" in str(self._device) else "?"
            print(f"[MadronaTiledCamera] gpu_id={gpu_id_str} waiting at post-init barrier...")
            torch.distributed.barrier()
            print(f"[MadronaTiledCamera] gpu_id={gpu_id_str} barrier passed, all ranks ready")

    def _init_madrona_renderer(self, geometry: MadronaGeometry):
        """Initialize Madrona BatchRenderer with geometry."""
        from madrona_mjx._madrona_mjx_batch_renderer import MadronaBatchRenderer
        from madrona_mjx.renderer import _setup_jax_primitives

        num_geoms = self._scene_adapter.num_geoms
        num_meshes = len(geometry.geom_vertex_offsets)

        # Build mesh data arrays
        mesh_vertices = np.ascontiguousarray(geometry.vertices, dtype=np.float32)
        mesh_faces = np.ascontiguousarray(geometry.indices, dtype=np.int32)

        assert mesh_vertices.ndim == 2 and mesh_vertices.shape[1] == 3
        assert mesh_faces.ndim == 2 and mesh_faces.shape[1] == 3

        mesh_vertex_offsets = np.ascontiguousarray(
            geometry.geom_vertex_offsets, dtype=np.int32
        )
        mesh_face_offsets = np.ascontiguousarray(
            geometry.geom_index_offsets, dtype=np.int32
        )

        # Texture coordinates
        if len(geometry.uvs) > 0:
            # Flip V coordinate (1 - v) to match Madrona's texture sampling convention
            uvs_flipped = geometry.uvs.copy()
            uvs_flipped[:, 1] = 1.0 - uvs_flipped[:, 1]
            mesh_texcoords = np.ascontiguousarray(uvs_flipped, dtype=np.float32)
            mesh_texcoord_num = []
            for i in range(num_meshes):
                if i < num_meshes - 1:
                    uv_count = (
                        geometry.geom_uv_offsets[i + 1] - geometry.geom_uv_offsets[i]
                    )
                else:
                    uv_count = len(geometry.uvs) - geometry.geom_uv_offsets[i]
                mesh_texcoord_num.append(uv_count)
            mesh_texcoord_offsets = np.ascontiguousarray(
                geometry.geom_uv_offsets, dtype=np.int32
            )
            mesh_texcoord_num = np.ascontiguousarray(mesh_texcoord_num, dtype=np.int32)
        else:
            mesh_texcoords = np.ascontiguousarray(np.zeros((0, 2), dtype=np.float32))
            mesh_texcoord_offsets = np.ascontiguousarray(
                np.zeros(num_meshes, dtype=np.int32)
            )
            mesh_texcoord_num = np.ascontiguousarray(
                np.zeros(num_meshes, dtype=np.int32)
            )

        # Geometry attributes
        geom_types = np.ascontiguousarray(np.full(num_geoms, 7, dtype=np.int32))
        geom_groups = np.ascontiguousarray(np.zeros(num_geoms, dtype=np.int32))
        geom_data_ids = np.ascontiguousarray(np.arange(num_geoms, dtype=np.int32))
        geom_sizes = np.ascontiguousarray(np.ones((num_geoms, 3), dtype=np.float32))

        if len(geometry.colors) == num_geoms:
            geom_rgba = np.ascontiguousarray(geometry.colors, dtype=np.float32)
        else:
            geom_rgba = np.ascontiguousarray(np.ones((num_geoms, 4), dtype=np.float32))

        # Textures and materials
        if geometry.textures and len(geometry.textures) > 0:
            all_tex_data = []
            tex_offsets = [0]
            tex_widths = []
            tex_heights = []
            tex_nchans = []

            # Match upstream madrona_mjx (renderer.py:166-172) exactly:
            #   - pack RGB bytes per texture into a flat array
            #   - tex_offsets are RGB-byte cumulative offsets (3 bytes/pixel)
            #   - after concat, do ONE global np.insert to expand to RGBA
            # mgr.cpp:569 then rebases RGB-byte offsets into the RGBA-expanded
            # buffer via `texOffsets[i] + texOffsets[i]/3 = (4/3)*RGB_off`. If we
            # passed RGBA bytes here, that rebase would double-shift every
            # texture except id 0 (offset 0 is a fixed point).
            for tex in geometry.textures:
                pixels = tex.pixels
                if pixels.shape[-1] == 4:
                    tex_rgb = pixels[:, :, :3].flatten()
                elif pixels.shape[-1] == 3:
                    tex_rgb = pixels.flatten()
                else:
                    tex_rgb = np.repeat(pixels.flatten(), 3)

                all_tex_data.append(tex_rgb)
                tex_offsets.append(tex_offsets[-1] + tex_rgb.size)
                tex_widths.append(tex.width)
                tex_heights.append(tex.height)
                tex_nchans.append(3)

            tex_data_rgb = np.concatenate(all_tex_data)
            tex_data = np.ascontiguousarray(
                np.insert(
                    tex_data_rgb,
                    np.arange(3, tex_data_rgb.shape[0], 3),
                    255,
                    axis=0,
                ),
                dtype=np.uint8,
            )
            tex_offsets = np.ascontiguousarray(tex_offsets[:-1], dtype=np.int32)
            tex_widths = np.ascontiguousarray(tex_widths, dtype=np.int32)
            tex_heights = np.ascontiguousarray(tex_heights, dtype=np.int32)
            tex_nchans = np.ascontiguousarray(tex_nchans, dtype=np.int32)

            num_materials = len(geometry.textures)
            mat_rgba = np.ascontiguousarray(
                np.ones((num_materials, 4), dtype=np.float32)
            )
            mat_tex_ids = np.full((num_materials, 10), -1, dtype=np.int32)
            for i in range(num_materials):
                mat_tex_ids[i, 1] = i
            mat_tex_ids = np.ascontiguousarray(mat_tex_ids)
            geom_mat_ids = np.ascontiguousarray(
                geometry.geom_texture_ids, dtype=np.int32
            )
        else:
            mat_rgba = np.ascontiguousarray(
                np.array([[0.8, 0.8, 0.8, 1.0]], dtype=np.float32)
            )
            mat_tex_ids = np.ascontiguousarray(np.full((1, 10), -1, dtype=np.int32))
            tex_data = np.ascontiguousarray(np.zeros(4, dtype=np.uint8))
            tex_offsets = np.ascontiguousarray(np.array([0], dtype=np.int32))
            tex_widths = np.ascontiguousarray(np.array([1], dtype=np.int32))
            tex_heights = np.ascontiguousarray(np.array([1], dtype=np.int32))
            tex_nchans = np.ascontiguousarray(np.array([4], dtype=np.int32))
            geom_mat_ids = np.ascontiguousarray(np.full(num_geoms, -1, dtype=np.int32))

        num_lights = max(1, self._scene_adapter.num_lights)

        # Camera FOV - one entry per camera, derived from the same physical
        # pinhole model IsaacLab uses for USD cameras.
        cam_fovy = np.ascontiguousarray(
            [self._compute_fovy_degrees(spawn) for spawn in self._camera_spawns],
            dtype=np.float32,
        )

        enabled_geom_groups = np.ascontiguousarray(np.array([0, 1, 2], dtype=np.int32))

        gpu_id = 0
        if "cuda" in str(self._device):
            try:
                gpu_id = int(str(self._device).split(":")[-1])
            except (ValueError, IndexError):
                gpu_id = 0

        print(
            f"[MadronaTiledCamera] num_geoms={num_geoms}, num_cams={self._num_cams}, gpu_id={gpu_id}"
        )
        if _FORCE_COLOR_OVERRIDES:
            print(
                "[MadronaTiledCamera] Material override diagnostics enabled: "
                "texture/material extraction is bypassed for raster RGB output."
            )

        try:
            # Serialize the ENTIRE Madrona init (renderer + JAX primitives +
            # CUDA flush) across processes.  The old lock only covered
            # MadronaBatchRenderer construction but released before
            # _setup_jax_primitives, allowing concurrent JIT/CUDA work on
            # another rank while the previous rank was still compiling.
            # This caused heap corruption (malloc unaligned tcache) and
            # distributed sync deadlocks at non-default resolutions.
            import fcntl

            # Read lock path from env var so the orchestrator can pin all
            # parallel slots to the same shared flock file.  Using
            # tempfile.gettempdir() here was wrong: the orchestrator sets
            # per-slot TMPDIR=/tmp/isaaclab_robust_slot_N, so each slot
            # would resolve a different path and the lock would not serialize.
            lock_path = os.environ.get("MADRONA_INIT_LOCK", "/tmp/madrona_init.lock")
            print(f"[MadronaTiledCamera] gpu_id={gpu_id} acquiring init lock...")
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                print(f"[MadronaTiledCamera] gpu_id={gpu_id} lock acquired, creating renderer")
                self._madrona_renderer = MadronaBatchRenderer(
                    gpu_id=gpu_id,
                    mesh_vertices=mesh_vertices,
                    mesh_faces=mesh_faces,
                    mesh_vertex_offsets=mesh_vertex_offsets,
                    mesh_face_offsets=mesh_face_offsets,
                    mesh_texcoords=mesh_texcoords,
                    mesh_texcoord_offsets=mesh_texcoord_offsets,
                    mesh_texcoord_num=mesh_texcoord_num,
                    geom_types=geom_types,
                    geom_groups=geom_groups,
                    geom_data_ids=geom_data_ids,
                    geom_sizes=geom_sizes,
                    geom_mat_ids=geom_mat_ids,
                    geom_rgba=geom_rgba,
                    mat_rgba=mat_rgba,
                    mat_tex_ids=mat_tex_ids,
                    tex_data=tex_data,
                    tex_offsets=tex_offsets,
                    tex_widths=tex_widths,
                    tex_heights=tex_heights,
                    tex_nchans=tex_nchans,
                    num_lights=num_lights,
                    num_cams=self._num_cams,  # Parameterized for multi-cam support
                    num_worlds=self._num_envs,
                    batch_render_view_width=self.cfg.width,
                    batch_render_view_height=self.cfg.height,
                    cam_fovy=cam_fovy,
                    enabled_geom_groups=enabled_geom_groups,
                    add_cam_debug_geo=False,
                    use_rt=not getattr(self.cfg, "use_rasterizer", True),
                )
                print(f"[MadronaTiledCamera] gpu_id={gpu_id} renderer created, setting up JAX primitives")

                self._init_fn, self._render_fn = _setup_jax_primitives(
                    self._madrona_renderer,
                    num_worlds=self._num_envs,
                    num_geoms=num_geoms,
                    num_cams=self._num_cams,
                    render_width=self.cfg.width,
                    render_height=self.cfg.height,
                )

                # Drain all async GPU work before releasing the lock.
                # Without this, in-flight CUDA kernels (shader compilation,
                # memory allocations) from this rank overlap with the next
                # rank's MadronaBatchRenderer construction, causing heap
                # corruption under VRAM pressure.
                torch.cuda.synchronize()
                print(f"[MadronaTiledCamera] gpu_id={gpu_id} init complete, releasing lock")

            self._num_geoms = num_geoms
            self._num_lights = num_lights
            self._geom_rgba = geom_rgba
            self._geom_sizes = geom_sizes
            self._geom_mat_ids = geom_mat_ids

            self._madrona_initialized = True
        except Exception as e:
            print(f"[MadronaTiledCamera] Failed to initialize Madrona: {e}")
            import traceback

            traceback.print_exc()
            self._madrona_initialized = False
            self._madrona_renderer = None

    @staticmethod
    def _normalize_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
        return quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)

    @staticmethod
    def _quat_xyzw_to_wxyz(quat: torch.Tensor) -> torch.Tensor:
        return quat[..., [3, 0, 1, 2]]

    @staticmethod
    def _env_uses_newton_quat_order(env) -> bool:
        sim_cfg = getattr(getattr(env, "cfg", None), "sim", None)
        return getattr(sim_cfg, "newton_cfg", None) is not None

    @staticmethod
    def _matrix_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
        quat = MadronaTiledCamera._normalize_quat_wxyz(quat)
        w, x, y, z = torch.unbind(quat, -1)
        two_s = 2.0 / (quat * quat).sum(-1)
        matrix = torch.stack(
            (
                1 - two_s * (y * y + z * z),
                two_s * (x * y - z * w),
                two_s * (x * z + y * w),
                two_s * (x * y + z * w),
                1 - two_s * (x * x + z * z),
                two_s * (y * z - x * w),
                two_s * (x * z - y * w),
                two_s * (y * z + x * w),
                1 - two_s * (x * x + y * y),
            ),
            dim=-1,
        )
        return matrix.reshape(quat.shape[:-1] + (3, 3))

    @staticmethod
    def _quat_wxyz_from_matrix(matrix: torch.Tensor) -> torch.Tensor:
        batch_shape = matrix.shape[:-2]
        flat = matrix.reshape(-1, 3, 3)
        quats = []
        for rot in flat:
            m00, m01, m02 = rot[0]
            m10, m11, m12 = rot[1]
            m20, m21, m22 = rot[2]
            trace = m00 + m11 + m22
            if trace > 0.0:
                s = torch.sqrt(trace + 1.0) * 2.0
                quat = torch.stack(
                    (
                        0.25 * s,
                        (m21 - m12) / s,
                        (m02 - m20) / s,
                        (m10 - m01) / s,
                    )
                )
            elif m00 > m11 and m00 > m22:
                s = torch.sqrt(1.0 + m00 - m11 - m22) * 2.0
                quat = torch.stack(
                    (
                        (m21 - m12) / s,
                        0.25 * s,
                        (m01 + m10) / s,
                        (m02 + m20) / s,
                    )
                )
            elif m11 > m22:
                s = torch.sqrt(1.0 + m11 - m00 - m22) * 2.0
                quat = torch.stack(
                    (
                        (m02 - m20) / s,
                        (m01 + m10) / s,
                        0.25 * s,
                        (m12 + m21) / s,
                    )
                )
            else:
                s = torch.sqrt(1.0 + m22 - m00 - m11) * 2.0
                quat = torch.stack(
                    (
                        (m10 - m01) / s,
                        (m02 + m20) / s,
                        (m12 + m21) / s,
                        0.25 * s,
                    )
                )
            quats.append(quat)
        quat_tensor = torch.stack(quats, dim=0).reshape(batch_shape + (4,))
        quat_tensor = MadronaTiledCamera._normalize_quat_wxyz(quat_tensor)
        return torch.where(quat_tensor[..., 0:1] < 0.0, -quat_tensor, quat_tensor)

    @staticmethod
    def _world_opengl_correction_matrix(
        device: torch.device | str, dtype: torch.dtype
    ) -> torch.Tensor:
        return torch.tensor(
            ((0.0, 0.0, -1.0), (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            device=device,
            dtype=dtype,
        )

    def _convert_camera_quat_wxyz(
        self,
        quat_wxyz: torch.Tensor,
        *,
        origin: str,
        target: str,
    ) -> torch.Tensor:
        """Convert camera-frame convention while preserving wxyz storage."""
        if origin == target:
            out = quat_wxyz.clone()
        else:
            if origin == "ros":
                rotm = self._matrix_from_quat_wxyz(quat_wxyz)
                rotm[:, :, 2] = -rotm[:, :, 2]
                rotm[:, :, 1] = -rotm[:, :, 1]
                quat_gl = self._quat_wxyz_from_matrix(rotm)
            elif origin == "world":
                rotm = self._matrix_from_quat_wxyz(quat_wxyz)
                correction = self._world_opengl_correction_matrix(
                    quat_wxyz.device, quat_wxyz.dtype
                )
                quat_gl = self._quat_wxyz_from_matrix(torch.matmul(rotm, correction))
            else:
                quat_gl = quat_wxyz

            if target == "ros":
                rotm = self._matrix_from_quat_wxyz(quat_gl)
                rotm[:, :, 2] = -rotm[:, :, 2]
                rotm[:, :, 1] = -rotm[:, :, 1]
                out = self._quat_wxyz_from_matrix(rotm)
            elif target == "world":
                rotm = self._matrix_from_quat_wxyz(quat_gl)
                correction = self._world_opengl_correction_matrix(
                    quat_wxyz.device, quat_wxyz.dtype
                )
                out = self._quat_wxyz_from_matrix(torch.matmul(rotm, correction.T))
            else:
                out = quat_gl.clone()

        return self._normalize_quat_wxyz(out)

    def _parse_prim_path_body(self, prim_path: str, scene_articulations: dict) -> tuple[str | None, str | None]:
        """Try to parse body_name and asset_name from a PhysX-style prim_path.

        Returns (body_name, asset_name) if parseable, (None, None) otherwise.
        Pattern: .../<asset_root>/<body_name>/<cam_name> where asset_root
        (lowercased) matches a scene articulation key.
        """
        segments = [s for s in prim_path.split("/") if s and s != "*"]
        if len(segments) < 3:
            return None, None
        cam_name = segments[-1]  # noqa: F841
        body_name = segments[-2]
        asset_root = segments[-3]
        asset_name_candidate = asset_root.lower()
        if asset_name_candidate in scene_articulations:
            return body_name, asset_name_candidate
        return None, None

    def _init_camera_poses(self):
        """Initialize camera poses from config offsets and environment origins.

        Builds pose tensors with shape (num_envs, num_cams, 3) for positions
        and (num_envs, num_cams, 4) for quaternions. Dynamic cameras (those
        following articulation body links) are identified and their bookkeeping
        arrays are stored for per-frame recomputation.
        """
        env = self._get_env()
        env_origins = env.scene.env_origins  # (num_envs, 3)
        scene_articulations = env.scene.articulations

        positions = []
        world_quaternions = []
        opengl_quaternions = []

        dyn_cam_indices = []
        dyn_body_names = []
        dyn_asset_names = []
        dyn_offset_pos_list = []
        dyn_offset_quat_opengl_list = []

        for cam_idx, (offset_pos, offset_rot, convention, cfg_body_name, cfg_asset_name) in enumerate(
            self._camera_offsets
        ):
            if convention not in ("world", "opengl", "ros"):
                raise ValueError(
                    f"Unsupported camera convention for Madrona camera {cam_idx}: {convention!r}. "
                    "Expected one of: 'world', 'opengl', 'ros'."
                )

            pos_tensor = torch.tensor(offset_pos, device=self._device, dtype=torch.float32)
            rot_tensor = torch.tensor(offset_rot, device=self._device, dtype=torch.float32)

            raw_quat_wxyz = rot_tensor.unsqueeze(0)

            body_name = cfg_body_name
            asset_name = cfg_asset_name

            if body_name is None:
                prim_path = getattr(self.cfg, "prim_path", None)
                if prim_path is not None:
                    parsed_body, parsed_asset = self._parse_prim_path_body(prim_path, scene_articulations)
                    if parsed_body is not None:
                        body_name = parsed_body
                        asset_name = parsed_asset

            if body_name is not None:
                if asset_name not in scene_articulations:
                    print(
                        f"[MadronaTiledCamera] cam {cam_idx}: asset '{asset_name}' not found "
                        f"in scene.articulations; treating camera as static."
                    )
                    body_name = None
                else:
                    articulation = scene_articulations[asset_name]
                    body_names_list = articulation.data.body_names
                    if body_name not in body_names_list:
                        raise ValueError(
                            f"MadronaTiledCamera cam {cam_idx}: body '{body_name}' not found "
                            f"in articulation '{asset_name}'. Available bodies: {body_names_list}"
                        )

            if body_name is not None:
                opengl_quat = self._convert_camera_quat_wxyz(
                    raw_quat_wxyz, origin=convention, target="opengl"
                )

                dyn_cam_indices.append(cam_idx)
                dyn_body_names.append(body_name)
                dyn_asset_names.append(asset_name)
                dyn_offset_pos_list.append(pos_tensor)
                dyn_offset_quat_opengl_list.append(opengl_quat.squeeze(0))

                placeholder_pos = env_origins.clone()
                world_quat_placeholder = self._convert_camera_quat_wxyz(
                    raw_quat_wxyz, origin=convention, target="world"
                )
                positions.append(placeholder_pos)
                world_quaternions.append(world_quat_placeholder.expand(self._num_envs, -1).clone())
                opengl_quaternions.append(opengl_quat.expand(self._num_envs, -1).clone())
            else:
                # Static camera: position = env_origin + offset_pos
                cam_pos = env_origins + pos_tensor.unsqueeze(0)
                world_quat = self._convert_camera_quat_wxyz(
                    raw_quat_wxyz, origin=convention, target="world"
                )
                opengl_quat = self._convert_camera_quat_wxyz(
                    raw_quat_wxyz, origin=convention, target="opengl"
                )

                positions.append(cam_pos)
                world_quaternions.append(world_quat.expand(self._num_envs, -1).clone())
                opengl_quaternions.append(opengl_quat.expand(self._num_envs, -1).clone())

        # Stack to (num_envs, num_cams, 3/4)
        self._static_camera_pos = torch.stack(positions, dim=1)
        self._static_camera_quat_world = torch.stack(world_quaternions, dim=1)
        self._static_camera_quat_opengl = torch.stack(opengl_quaternions, dim=1)

        self._dyn_camera_mask = dyn_cam_indices
        self._dyn_asset_names = dyn_asset_names
        self._dyn_body_names_list = dyn_body_names

        if dyn_cam_indices:
            grouped: dict[str, list[tuple[int, int, int]]] = {}
            for local_i, (cam_i, b_name, a_name) in enumerate(
                zip(dyn_cam_indices, dyn_body_names, dyn_asset_names)
            ):
                articulation = scene_articulations[a_name]
                body_names_list = articulation.data.body_names
                b_idx = body_names_list.index(b_name)
                grouped.setdefault(a_name, []).append((cam_i, b_idx, local_i))
            self._dyn_grouped: dict[str, list[tuple[int, int, int]]] = grouped

            self._dyn_offset_pos = torch.stack(dyn_offset_pos_list, dim=0)
            self._dyn_offset_quat_opengl_wxyz = torch.stack(dyn_offset_quat_opengl_list, dim=0)

            if dyn_cam_indices:
                print(
                    f"[MadronaTiledCamera] {len(dyn_cam_indices)} dynamic camera(s): "
                    + ", ".join(
                        f"cam{ci}->'{bn}'@'{an}'"
                        for ci, bn, an in zip(dyn_cam_indices, dyn_body_names, dyn_asset_names)
                    )
                )

    def _init_buffer_pool(self):
        """Initialize persistent buffer pool for zero-copy GPU memory sharing.

        Pre-allocates buffers once and creates persistent JAX views via DLPack.
        This eliminates per-frame tensor allocation and DLPack conversion overhead.
        """
        num_geoms = self._scene_adapter.num_geoms
        num_lights = max(1, self._scene_adapter.num_lights)

        self._buffer_pool = PersistentBufferPool(self._device)

        # Register state buffers
        self._buffer_pool.register(
            "geom_positions", BufferSpec(shape=(self._num_envs, num_geoms, 3))
        )
        self._buffer_pool.register(
            "geom_quaternions", BufferSpec(shape=(self._num_envs, num_geoms, 4))
        )
        self._buffer_pool.register(
            "geom_scales", BufferSpec(shape=(self._num_envs, num_geoms, 3))
        )
        self._buffer_pool.register(
            "camera_positions", BufferSpec(shape=(self._num_envs, self._num_cams, 3))
        )
        self._buffer_pool.register(
            "camera_quaternions", BufferSpec(shape=(self._num_envs, self._num_cams, 4))
        )
        self._buffer_pool.register(
            "light_positions", BufferSpec(shape=(self._num_envs, num_lights, 3))
        )
        self._buffer_pool.register(
            "light_directions", BufferSpec(shape=(self._num_envs, num_lights, 3))
        )

        # Initialize all buffers (one-time DLPack conversion cost)
        self._buffer_pool.initialize()

    def _recompute_dynamic_camera_poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute world poses for dynamic cameras from live articulation state.

        Returns composed (camera_pos, camera_quat_opengl) tensors of shape
        (num_envs, num_cams, 3/4) ready to pass to build_state_inplace.
        Static camera slots are copied from the cached static tensors.
        """
        from isaaclab.utils.math import combine_frame_transforms

        cam_pos = self._static_camera_pos.clone()
        cam_quat = self._static_camera_quat_opengl.clone()

        env = self._get_env()
        scene_articulations = env.scene.articulations

        for asset_name, entries in self._dyn_grouped.items():
            articulation = scene_articulations[asset_name]
            body_indices_t = torch.tensor(
                [e[1] for e in entries], device=self._device, dtype=torch.long
            )
            body_pos_w, body_quat_w_wxyz = self._scene_adapter._gather_articulation_body_poses(
                articulation, body_indices_t
            )

            for slot, (cam_i, b_idx, local_idx) in enumerate(entries):
                link_pos = body_pos_w[:, slot, :]  # (num_envs, 3)
                link_quat = body_quat_w_wxyz[:, slot, :]  # (num_envs, 4)

                offset_pos = self._dyn_offset_pos[local_idx].unsqueeze(0).expand(self._num_envs, -1)
                offset_quat = self._dyn_offset_quat_opengl_wxyz[local_idx].unsqueeze(0).expand(self._num_envs, -1)

                world_cam_pos, world_cam_quat = combine_frame_transforms(
                    link_pos, link_quat, offset_pos, offset_quat
                )

                cam_pos[:, cam_i, :] = world_cam_pos
                cam_quat[:, cam_i, :] = world_cam_quat

        self._data.pos_w = cam_pos[:, 0, :].clone()
        self._data.info["pos_w_by_camera"] = cam_pos.clone()
        self._data.info["quat_w_opengl_by_camera"] = cam_quat.clone()

        return cam_pos, cam_quat

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Render new frame for outdated environments."""
        if not self._madrona_initialized:
            env = _ENV_REGISTRY.get("current")
            if env is not None:
                self._lazy_init_madrona()
            else:
                return

        self._frame[env_ids] += 1

        if self._scene_adapter is None or self._buffer_pool is None:
            self._fill_placeholder_output()
            return

        # Get marker positions from command manager
        if _PROFILE_ENABLED:
            t0 = time.perf_counter()

        marker_positions = {}
        marker_quaternions = {}
        env = self._get_env()
        if hasattr(env, "command_manager"):
            try:
                cmd = env.command_manager.get_term("object_pose")
                if (
                    cmd is not None
                    and hasattr(cmd, "pos_command_w")
                    and hasattr(cmd, "quat_command_w")
                ):
                    marker_pos_offset = getattr(
                        cmd.cfg, "marker_pos_offset", (0.0, 0.0, 0.0)
                    )
                    offset_tensor = torch.tensor(
                        marker_pos_offset, device=self._device
                    )
                    visual_marker_pos = cmd.pos_command_w + offset_tensor

                    for mapping in self._scene_adapter._geom_mappings:
                        if mapping.asset_type == "marker":
                            marker_positions[mapping.asset_name] = visual_marker_pos
                            marker_quat = cmd.quat_command_w
                            if self._env_uses_newton_quat_order(env):
                                marker_quat = self._quat_xyzw_to_wxyz(marker_quat)
                            marker_quaternions[mapping.asset_name] = marker_quat
            except Exception:
                pass

        if _PROFILE_ENABLED:
            torch.cuda.synchronize()
            _profiler.record("1_marker_lookup", time.perf_counter() - t0)
            t0 = time.perf_counter()

        if self._dyn_camera_mask:
            camera_positions, camera_quaternions = self._recompute_dynamic_camera_poses()
        else:
            camera_positions = self._static_camera_pos
            camera_quaternions = self._static_camera_quat_opengl

        # Use optimized in-place state update (writes directly to pre-allocated buffers)
        self._scene_adapter.build_state_inplace(
            geom_positions_out=self._buffer_pool.get_torch("geom_positions"),
            geom_quaternions_out=self._buffer_pool.get_torch("geom_quaternions"),
            geom_scales_out=self._buffer_pool.get_torch("geom_scales"),
            cam_pos_out=self._buffer_pool.get_torch("camera_positions"),
            cam_quat_out=self._buffer_pool.get_torch("camera_quaternions"),
            light_positions_out=self._buffer_pool.get_torch("light_positions"),
            light_directions_out=self._buffer_pool.get_torch("light_directions"),
            camera_positions=camera_positions,
            camera_quaternions=camera_quaternions,
            marker_positions=marker_positions,
            marker_quaternions=marker_quaternions,
            camera_convention="opengl",
        )

        if _PROFILE_ENABLED:
            torch.cuda.synchronize()
            _profiler.record("2_build_state", time.perf_counter() - t0)

        try:
            rgb, depth = self._render_madrona_optimized()
            self._store_render_output(rgb, depth)

            if self._render_goal_pass:
                # Done after the primary pass: clobbers the geom buffers with
                # FK-derived poses for the robot and a far-away sentinel for
                # everything else, renders, stores result. The next frame's
                # build_state_inplace reseeds from live state, so no restore.
                self._do_goal_pass()

            if _PROFILE_ENABLED:
                _profiler.tick()

        except Exception as e:
            print(f"[MadronaTiledCamera] Render failed: {e}")
            import traceback

            traceback.print_exc()
            self._fill_placeholder_output()

    def _render_madrona_optimized(self):
        """Render using pre-allocated buffer pool (zero DLPack conversion cost).

        Uses persistent JAX views from the buffer pool that point to the same
        GPU memory as the PyTorch tensors. Since build_state_inplace already
        wrote to the PyTorch tensors, the JAX views see the updated data.
        """
        if _PROFILE_ENABLED:
            t0 = time.perf_counter()

        # Get pre-existing JAX views (no conversion needed - same GPU memory)
        geom_pos_jax = self._buffer_pool.get_jax("geom_positions")
        geom_quat_jax = self._buffer_pool.get_jax("geom_quaternions")
        cam_pos_jax = self._buffer_pool.get_jax("camera_positions")
        cam_quat_jax = self._buffer_pool.get_jax("camera_quaternions")

        if _PROFILE_ENABLED:
            _profiler.record("3a_get_jax_views", time.perf_counter() - t0)
            t0 = time.perf_counter()

        if self._needs_init:
            rgb_jax, depth_jax, self._render_token = self._call_init_optimized(
                geom_pos_jax, geom_quat_jax, cam_pos_jax, cam_quat_jax
            )
            self._needs_init = False
        else:
            rgb_jax, depth_jax, self._render_token = self._render_fn(
                self._render_token,
                geom_pos_jax,
                geom_quat_jax,
                cam_pos_jax,
                cam_quat_jax,
            )

        if _PROFILE_ENABLED:
            # Don't sync here - measure async dispatch time
            _profiler.record("3b_jax_render_dispatch", time.perf_counter() - t0)
            t0 = time.perf_counter()
            # Now sync to measure actual render time
            rgb_jax.block_until_ready()
            _profiler.record("3c_jax_render_sync", time.perf_counter() - t0)
            t0 = time.perf_counter()

        rgb_torch = self._jax_to_torch(rgb_jax)

        if _PROFILE_ENABLED:
            torch.cuda.synchronize()
            _profiler.record("3d_rgb_jax_to_torch", time.perf_counter() - t0)
            t0 = time.perf_counter()

        # Only convert depth if needed (skip for RGB-only configs)
        if self._needs_depth:
            depth_torch = self._jax_to_torch(depth_jax)
            if _PROFILE_ENABLED:
                torch.cuda.synchronize()
                _profiler.record("3e_depth_jax_to_torch", time.perf_counter() - t0)
        else:
            depth_torch = None

        return rgb_torch, depth_torch

    def _read_goal_qpos(self, env, robot) -> torch.Tensor:
        """Resolve the joint-position signal the goal pass should render.

        See MadronaTiledCameraCfg.goal_qpos_source for the two modes.
        Falls back to robot.data.joint_pos_target with a one-time warning if
        the action term cannot be located.
        """
        source = getattr(self.cfg, "goal_qpos_source", "target")

        if source == "target":
            return robot.data.joint_pos_target

        if source == "raw_action":
            term_name = self.cfg.goal_action_term_name
            try:
                term = env.action_manager.get_term(term_name)
            except Exception:
                if not getattr(self, "_warned_no_action_term", False):
                    print(
                        f"[MadronaTiledCamera] goal_qpos_source='raw_action' but "
                        f"action term '{term_name}' not found in action_manager; "
                        f"falling back to joint_pos_target."
                    )
                    self._warned_no_action_term = True
                return robot.data.joint_pos_target

            raw = getattr(term, "raw_actions", None)
            scaled = None

            # EMA-style action term (e.g. EMAJointPositionAction): linear scale
            # from raw [-1, 1] action to [joint_lower, joint_upper], BYPASSING
            # the EMA smoothing. This is what the policy commanded this step.
            j_lo = getattr(term, "_joint_lower", None)
            j_hi = getattr(term, "_joint_upper", None)
            if raw is not None and j_lo is not None and j_hi is not None:
                scaled = 0.5 * (raw.clamp(-1.0, 1.0) + 1.0) * (j_hi - j_lo) + j_lo

            # Vanilla isaaclab JointPositionAction: processed = raw*scale + offset
            # (line 172 of IsaacLab/.../mdp/actions/joint_actions.py). No EMA,
            # so this matches what the actuator was just told to target — same
            # value as `target` mode but bypassing actuator dynamics.
            if scaled is None and raw is not None:
                s = getattr(term, "_scale", None)
                o = getattr(term, "_offset", None)
                if s is not None and o is not None:
                    scaled = raw * s + o

            if scaled is None:
                if not getattr(self, "_warned_action_term_shape", False):
                    cls_name = type(term).__name__
                    print(
                        f"[MadronaTiledCamera] action term '{term_name}' "
                        f"({cls_name}) lacks both EMA (_joint_lower/_joint_upper) "
                        f"and vanilla (_scale/_offset) attrs; falling back to "
                        f"joint_pos_target."
                    )
                    self._warned_action_term_shape = True
                return robot.data.joint_pos_target

            joint_ids = getattr(term, "_joint_ids", None)
            if isinstance(joint_ids, slice) and joint_ids == slice(None):
                return scaled
            if joint_ids is None:
                return scaled
            target = robot.data.joint_pos_target.clone()
            target[:, joint_ids] = scaled
            return target

        if not getattr(self, "_warned_unknown_source", False):
            print(
                f"[MadronaTiledCamera] unknown goal_qpos_source={source!r}; "
                f"falling back to joint_pos_target."
            )
            self._warned_unknown_source = True
        return robot.data.joint_pos_target

    def _do_goal_pass(self):
        """Run the second render pass with the robot at its idealized pose.

        Hacks the existing geom buffers (no second pool): writes FK output for
        the robot articulation into its slots and a sentinel z into every
        other slot so non-robot geometry falls outside the frustum. Renders,
        copies result to data.output["goal_rgb"]. The next frame's
        build_state_inplace clobbers our writes with live sim state, which is
        why no restore is needed.
        """
        env = self._get_env()
        adapter = self._scene_adapter

        robot_name = self.cfg.goal_robot_asset_name
        robot = env.scene.articulations.get(robot_name)
        if robot is None:
            print(
                f"[MadronaTiledCamera] goal pass: no articulation '{robot_name}' "
                f"in scene; disabling goal pass."
            )
            self._render_goal_pass = False
            return

        if self._fk_chain is None:
            self._fk_chain = USDArticulationChain(robot, device=self._device)
            print(
                f"[MadronaTiledCamera] FK chain built for '{robot_name}': "
                f"{self._fk_chain.num_bodies} bodies, "
                f"{self._fk_chain.num_dofs} dofs, "
                f"missing_joint_bodies={self._fk_chain._missing_joint_bodies}"
            )

        # Ensure adapter slot tables are built (normally lazy on first
        # build_state_inplace; that ran above so this is a no-op).
        if not adapter._geom_lookup_built:
            adapter._build_geom_lookup_tables()

        if not hasattr(self, "_goal_robot_geom_idx_t"):
            geom_slots = adapter._asset_geom_ranges.get(robot_name, [])
            body_slots = adapter._asset_body_indices.get(robot_name, [])
            if not geom_slots:
                print(
                    f"[MadronaTiledCamera] goal pass: '{robot_name}' has no geom "
                    f"slots in adapter; disabling goal pass."
                )
                self._render_goal_pass = False
                return
            self._goal_robot_geom_idx_t = torch.tensor(
                geom_slots, device=self._device, dtype=torch.long
            )
            self._goal_robot_body_idx_t = torch.tensor(
                body_slots, device=self._device, dtype=torch.long
            )

        # Pick the joint-position signal to render. Two interpretations live in
        # cfg.goal_qpos_source: "target" (post-EMA actuator setpoint, invariant
        # to actuator dynamics only) or "raw_action" (raw policy command put
        # through the action term's linear scale, invariant to actuator AND
        # action-term smoothing).
        target_qpos = self._read_goal_qpos(env, robot)
        if self._goal_qpos_lag_check_remaining > 0:
            try:
                err = (target_qpos - robot.data.joint_pos).abs().max().item()
                print(
                    f"[MadronaTiledCamera] goal_qpos_lag_check: "
                    f"max|joint_pos_target - joint_pos|={err:.4e}"
                )
            except Exception:
                pass
            self._goal_qpos_lag_check_remaining -= 1

        root_pose_w = robot.data.body_link_pose_w[:, 0]
        body_pos_w, body_quat_w = self._fk_chain.forward_kinematics(
            target_qpos, root_pose_w
        )

        # Madrona renders each env in its env-local frame; subtract origins.
        env_origins = env.scene.env_origins.unsqueeze(1)  # (E, 1, 3)
        body_pos_local = body_pos_w - env_origins

        geom_pos = self._buffer_pool.get_torch("geom_positions")
        geom_quat = self._buffer_pool.get_torch("geom_quaternions")

        # Push everything off camera, then stamp the robot.
        geom_pos[..., 0:2] = 0.0
        geom_pos[..., 2] = self.cfg.goal_geom_sentinel_z
        geom_quat[..., 0] = 1.0
        geom_quat[..., 1:] = 0.0

        geom_pos[:, self._goal_robot_geom_idx_t] = body_pos_local.index_select(
            1, self._goal_robot_body_idx_t
        )
        geom_quat[:, self._goal_robot_geom_idx_t] = body_quat_w.index_select(
            1, self._goal_robot_body_idx_t
        )

        rgb, _ = self._render_madrona_optimized()

        for cam_idx in self._resolve_goal_cam_indices():
            key = self._goal_output_key(cam_idx)
            self._data.output[key][:] = rgb[:, cam_idx].narrow(-1, 0, 3)

    def _call_init_optimized(
        self, geom_pos_jax, geom_quat_jax, cam_pos_jax, cam_quat_jax
    ):
        """Call Madrona init function using pre-allocated buffers where possible."""
        import jax

        # Get the JAX device matching our PyTorch device
        gpu_id = 0
        if "cuda" in str(self._device):
            try:
                gpu_id = int(str(self._device).split(":")[-1])
            except (ValueError, IndexError):
                gpu_id = 0

        # Find the matching JAX device
        jax_devices = jax.devices("gpu")
        if gpu_id < len(jax_devices):
            target_device = jax_devices[gpu_id]
        else:
            target_device = jax_devices[0]
            print(
                f"[MadronaTiledCamera] Warning: gpu_id={gpu_id} not found, using {target_device}"
            )

        render_token = jax.device_put(jnp.array((), dtype=jnp.bool_), target_device)

        # These are static data, build once and cache
        if not hasattr(self, "_cached_init_data"):
            geom_mat_ids_base = jnp.array(self._geom_mat_ids, dtype=jnp.int32)
            if _FORCE_COLOR_OVERRIDES:
                geom_mat_ids_base = jnp.full(
                    (self._num_geoms,), -2, dtype=jnp.int32
                )
            geom_mat_ids = jnp.repeat(
                jnp.expand_dims(geom_mat_ids_base, 0), self._num_envs, axis=0
            )

            def rgb_to_uint32(rgba):
                r = int(rgba[0] * 255)
                g = int(rgba[1] * 255)
                b = int(rgba[2] * 255)
                return (r << 16) | (g << 8) | b

            geom_rgb_packed_base = jnp.array(
                [rgb_to_uint32(self._geom_rgba[i]) for i in range(self._num_geoms)],
                dtype=jnp.uint32,
            )
            geom_rgb_packed = jnp.repeat(
                jnp.expand_dims(geom_rgb_packed_base, 0), self._num_envs, axis=0
            )

            geom_sizes_base = jnp.array(self._geom_sizes, dtype=jnp.float32)
            geom_sizes = jnp.repeat(
                jnp.expand_dims(geom_sizes_base, 0), self._num_envs, axis=0
            )

            light_positions = []
            light_directions = []
            for light in self._scene_adapter.lights:
                light_positions.append(light.position)
                light_directions.append(light.direction)

            if len(light_positions) == 0:
                light_positions = [[0.0, 10.0, 10.0]]
                light_directions = [[0.0, -0.7071, -0.7071]]

            light_pos_base = jnp.array(light_positions, dtype=jnp.float32)
            light_dir_base = jnp.array(light_directions, dtype=jnp.float32)
            light_pos = jnp.tile(light_pos_base, (self._num_envs, 1, 1))
            light_dir = jnp.tile(light_dir_base, (self._num_envs, 1, 1))

            # Per-light shadow flag from the adapter so DomeLight-approximation
            # directions (`casts_shadow=False`) match PhysX's ambient/dome behavior
            # instead of casting 6 separate hard shadows from the synthetic dome
            # directions. PhysX integrates the dome as an environment light --
            # no shadow contribution. With our 6 dome directions set to
            # `casts_shadow=False`, we get the same soft-ambient appearance.
            shadow_flags = [bool(light.casts_shadow) for light in self._scene_adapter.lights]
            if len(shadow_flags) == 0:
                shadow_flags = [True]
            # Pad/truncate to self._num_lights to match the buffer shape declared
            # at MadronaBatchRenderer construction.
            while len(shadow_flags) < self._num_lights:
                shadow_flags.append(True)
            shadow_flags = shadow_flags[: self._num_lights]
            light_isdir_base = jnp.array([True] * self._num_lights, dtype=jnp.bool_)
            light_castshadow_base = jnp.array(shadow_flags, dtype=jnp.bool_)
            light_cutoff_base = jnp.array([45.0] * self._num_lights, dtype=jnp.float32)
            light_isdir = jnp.tile(light_isdir_base, (self._num_envs, 1))
            light_castshadow = jnp.tile(light_castshadow_base, (self._num_envs, 1))
            light_cutoff = jnp.tile(light_cutoff_base, (self._num_envs, 1))

            # Cache the static data - put on target device to avoid GPU 0 memory bloat
            self._cached_init_data = {
                "geom_mat_ids": jax.device_put(geom_mat_ids, target_device),
                "geom_rgb_packed": jax.device_put(geom_rgb_packed, target_device),
                "geom_sizes": jax.device_put(geom_sizes, target_device),
                "light_pos": jax.device_put(light_pos, target_device),
                "light_dir": jax.device_put(light_dir, target_device),
                "light_isdir": jax.device_put(light_isdir, target_device),
                "light_castshadow": jax.device_put(light_castshadow, target_device),
                "light_cutoff": jax.device_put(light_cutoff, target_device),
            }

        cached = self._cached_init_data

        rgb, depth, render_token = self._init_fn(
            render_token,
            geom_pos_jax,
            geom_quat_jax,
            cam_pos_jax,
            cam_quat_jax,
            cached["geom_mat_ids"],
            cached["geom_rgb_packed"],
            cached["geom_sizes"],
            cached["light_pos"],
            cached["light_dir"],
            cached["light_isdir"],
            cached["light_castshadow"],
            cached["light_cutoff"],
        )

        return rgb, depth, render_token

    def _torch_to_jax(self, tensor: torch.Tensor):
        """Convert PyTorch tensor to JAX array via DLPack."""
        tensor = tensor.contiguous()
        return jax_dlpack.from_dlpack(to_dlpack(tensor))

    def _jax_to_torch(self, array):
        """Convert JAX array to PyTorch tensor via DLPack.

        Ensures the resulting tensor is on the correct device (self._device).
        DLPack should preserve the device, but we verify and move if needed.
        """
        tensor = from_dlpack(jax_dlpack.to_dlpack(array))
        # Ensure tensor is on the correct device - DLPack may not preserve device in multi-GPU
        # self._device may be a string or torch.device, so convert to torch.device for comparison
        expected_device = (
            torch.device(self._device)
            if isinstance(self._device, str)
            else self._device
        )
        tensor_device = tensor.device
        if (
            tensor_device.type != expected_device.type
            or tensor_device.index != expected_device.index
        ):
            tensor = tensor.to(expected_device)
        return tensor

    def _create_buffers(self):
        """Allocate output tensors with per-camera keys."""
        # Store first camera pose data for compatibility
        self._data.pos_w = self._static_camera_pos[:, 0, :].clone()
        self._data.quat_w_world = self._static_camera_quat_world[:, 0, :].clone()
        intrinsics_by_camera = torch.stack(
            [self._compute_intrinsics(cam_idx) for cam_idx in range(self._num_cams)],
            dim=1,
        )
        self._data.intrinsic_matrices = intrinsics_by_camera[:, 0].clone()
        self._data.image_shape = (self.cfg.height, self.cfg.width)

        output = {}
        for data_type in self.cfg.data_types:
            for cam_idx in range(self._num_cams):
                key = self._output_key(data_type, cam_idx)

                if data_type in ("rgb", "rgba"):
                    channels = 4 if data_type == "rgba" else 3
                    output[key] = torch.zeros(
                        (self._num_envs, self.cfg.height, self.cfg.width, channels),
                        device=self._device,
                        dtype=torch.uint8,
                    )
                elif data_type == "depth":
                    output[key] = torch.zeros(
                        (self._num_envs, self.cfg.height, self.cfg.width, 1),
                        device=self._device,
                        dtype=torch.float32,
                    )

        if self._render_goal_pass:
            for cam_idx in self._resolve_goal_cam_indices():
                key = self._goal_output_key(cam_idx)
                output[key] = torch.zeros(
                    (self._num_envs, self.cfg.height, self.cfg.width, 3),
                    device=self._device,
                    dtype=torch.uint8,
                )

        self._data.output = output
        self._data.info = {
            "intrinsic_matrices_by_camera": intrinsics_by_camera,
            "pos_w_by_camera": self._static_camera_pos.clone(),
            "quat_w_world_by_camera": self._static_camera_quat_world.clone(),
            "quat_w_opengl_by_camera": self._static_camera_quat_opengl.clone(),
        }

    def _fill_placeholder_output(self):
        """Fill output with placeholder data."""
        for key in self._data.output:
            if "rgb" in key or "rgba" in key:
                self._data.output[key].fill_(128)
            elif "depth" in key:
                self._data.output[key].fill_(1.0)

    def _store_render_output(self, rgb: torch.Tensor, depth: torch.Tensor | None):
        """Store Madrona output to per-camera keys.

        Madrona output:
            rgb: (num_worlds, num_cameras, height, width, 4) uint8
            depth: (num_worlds, num_cameras, height, width, 1) float32 or None

        Stored as:
            output["rgb"]: First camera (num_envs, H, W, 3)
            output["rgb_1"]: Second camera (num_envs, H, W, 3)
            etc.
        """
        if _PROFILE_ENABLED:
            t0 = time.perf_counter()

        # Vectorized copy for RGBA (no channel slicing needed - fastest path)
        if "rgba" in self.cfg.data_types:
            for cam_idx in range(self._num_cams):
                key = self._output_key("rgba", cam_idx)
                self._data.output[key][:] = rgb[:, cam_idx]

        # RGB requires channel slicing RGBA->RGB (creates copy)
        # Use narrow() instead of [..., :3] for potentially better performance
        if "rgb" in self.cfg.data_types:
            for cam_idx in range(self._num_cams):
                key = self._output_key("rgb", cam_idx)
                # narrow(dim, start, length) is more efficient than slice
                self._data.output[key][:] = rgb[:, cam_idx].narrow(-1, 0, 3)

        if "depth" in self.cfg.data_types and depth is not None:
            for cam_idx in range(self._num_cams):
                key = self._output_key("depth", cam_idx)
                self._data.output[key][:] = depth[:, cam_idx]

        if _PROFILE_ENABLED:
            torch.cuda.synchronize()
            _profiler.record("4a_copy_to_output", time.perf_counter() - t0)

        # Save first image for debugging if enabled
        if _SAVE_FIRST_IMAGE and not MadronaTiledCamera._first_image_saved:
            self._save_first_image_to_file(rgb, depth)

    def _save_first_image_to_file(self, rgb: torch.Tensor, depth: torch.Tensor | None):
        """Save the first rendered image to file for debugging.

        Called only once on the first render when MADRONA_SAVE_FIRST_IMAGE=1.
        Saves RGB and depth images from the first environment for all cameras.

        Args:
            rgb: RGB output tensor (num_worlds, num_cameras, height, width, 4) uint8
            depth: Depth output tensor (num_worlds, num_cameras, height, width, 1) or None
        """
        try:
            import cv2

            debug_dir = "tmp/madrona_debug"
            os.makedirs(debug_dir, exist_ok=True)

            # rgb shape: (num_worlds, num_cams, H, W, 4) or (num_worlds, num_cams, H, W, 3)
            num_cams = rgb.shape[1]

            # Save RGB from first env, all cameras
            for cam_idx in range(num_cams):
                rgb_img = rgb[0, cam_idx].cpu().numpy()  # First env, camera cam_idx
                if rgb_img.shape[-1] == 4:
                    rgb_img = rgb_img[:, :, :3]  # Remove alpha
                # Convert RGB to BGR for OpenCV
                suffix = f"_cam{cam_idx}" if num_cams > 1 else ""
                cv2.imwrite(f"{debug_dir}/first_rgb{suffix}.png", rgb_img[:, :, ::-1])
                print(
                    f"[MadronaTiledCamera] Saved first RGB image to {debug_dir}/first_rgb{suffix}.png"
                )

            # Save depth from first env, all cameras if available
            if depth is not None:
                for cam_idx in range(num_cams):
                    depth_img = (
                        depth[0, cam_idx, :, :, 0].cpu().numpy()
                    )  # First env, camera cam_idx
                    # Normalize depth for visualization
                    depth_finite = depth_img[np.isfinite(depth_img)]
                    if len(depth_finite) > 0:
                        d_min, d_max = depth_finite.min(), depth_finite.max()
                        depth_vis = (depth_img - d_min) / (d_max - d_min + 1e-6)
                        depth_vis = (depth_vis * 255).astype(np.uint8)
                    else:
                        depth_vis = np.zeros_like(depth_img, dtype=np.uint8)
                    suffix = f"_cam{cam_idx}" if num_cams > 1 else ""
                    cv2.imwrite(f"{debug_dir}/first_depth{suffix}.png", depth_vis)
                    print(
                        f"[MadronaTiledCamera] Saved first depth image to {debug_dir}/first_depth{suffix}.png"
                    )

            # Mark as saved so we don't save again
            MadronaTiledCamera._first_image_saved = True

        except ImportError:
            print("[MadronaTiledCamera] Could not save first image (cv2 not available)")
        except Exception as e:
            print(f"[MadronaTiledCamera] Failed to save first image: {e}")
            # Still mark as saved to avoid repeated failures
            MadronaTiledCamera._first_image_saved = True

    def _validate_pinhole_spawn(self, spawn, cam_idx: int) -> None:
        """Reject camera settings Madrona's FOV-only projection cannot represent."""
        if spawn is None:
            raise ValueError(f"Madrona camera {cam_idx} requires a pinhole spawn config.")
        if "Fisheye" in type(spawn).__name__:
            raise ValueError(
                f"Madrona camera {cam_idx} requires a pinhole camera, "
                f"got {type(spawn).__name__}."
            )

        horizontal_offset = getattr(spawn, "horizontal_aperture_offset", 0.0)
        vertical_offset = getattr(spawn, "vertical_aperture_offset", 0.0)
        if abs(horizontal_offset) > 1e-12 or abs(vertical_offset) > 1e-12:
            raise ValueError(
                f"Madrona camera {cam_idx} requires centered pinhole intrinsics; "
                f"got aperture offsets ({horizontal_offset}, {vertical_offset})."
            )

        focal_length = getattr(spawn, "focal_length", None)
        horizontal_aperture = getattr(spawn, "horizontal_aperture", None)
        if focal_length is None or focal_length <= 0.0:
            raise ValueError(f"Madrona camera {cam_idx} requires a positive focal_length.")
        if horizontal_aperture is None or horizontal_aperture <= 0.0:
            raise ValueError(f"Madrona camera {cam_idx} requires a positive horizontal_aperture.")

        vertical_aperture = getattr(spawn, "vertical_aperture", None)
        if vertical_aperture is not None and vertical_aperture <= 0.0:
            raise ValueError(f"Madrona camera {cam_idx} vertical_aperture must be positive.")

    def _vertical_aperture(self, spawn) -> float:
        vertical_aperture = getattr(spawn, "vertical_aperture", None)
        if vertical_aperture is not None:
            return float(vertical_aperture)
        return float(spawn.horizontal_aperture) * self.cfg.height / self.cfg.width

    def _compute_fovy_degrees(self, spawn) -> float:
        vertical_aperture = self._vertical_aperture(spawn)
        return float(
            2.0 * np.degrees(np.arctan(vertical_aperture / (2.0 * spawn.focal_length)))
        )

    def _compute_intrinsics(self, cam_idx: int = 0) -> torch.Tensor:
        """Compute camera intrinsic matrices from physical pinhole settings."""
        spawn = self._camera_spawns[cam_idx]
        vertical_aperture = self._vertical_aperture(spawn)
        fx = spawn.focal_length * self.cfg.width / spawn.horizontal_aperture
        fy = spawn.focal_length * self.cfg.height / vertical_aperture

        cx = self.cfg.width / 2
        cy = self.cfg.height / 2

        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            device=self._device,
            dtype=torch.float32,
        )

        return K.unsqueeze(0).expand(self._num_envs, -1, -1).clone()

    def _get_env(self):
        """Get environment reference."""
        if hasattr(self, "_env") and self._env is not None:
            return self._env
        return _get_registered_env()

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization state."""
        pass

    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        pass

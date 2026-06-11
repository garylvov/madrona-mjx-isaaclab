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

"""Configuration for Madrona-backed tiled camera."""

from __future__ import annotations

from dataclasses import MISSING, field

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass


@configclass
class MadronaTiledCameraCfg(CameraCfg):
    """Configuration for Madrona-backed tiled camera.

    This camera uses GPU rasterization via Madrona for 10-100x faster
    rendering compared to RTX-based TiledCamera.

    Limitations:
        - Only supports rgb and depth output types
        - No ray-traced effects (reflections, refractions, realistic shadows)
        - No semantic/instance segmentation
        - Limited material support (basic colors only)
    """

    from .camera import MadronaTiledCamera

    class_type: type = MadronaTiledCamera

    data_types: list[str] = MISSING
    """Data types to capture. Supported: ["rgb", "rgba", "depth"]"""

    use_rasterizer: bool = True
    """Use rasterizer (True) or raytracer (False).

    Rasterizer is faster but raytracer supports shadows and better lighting.
    """

    add_debug_geometry: bool = False
    """Add debug geometry for cameras in the scene."""

    max_texture_size: int | None = None
    """Maximum texture edge (pixels) for every texture this renderer ingests.

    Textures are part of the geometry pool shared by all cameras of this
    renderer instance, so this is a per-renderer setting exposed at the camera
    config level. None falls back to the legacy MADRONA_MAX_TEXTURE_SIZE env
    var, then to the default (512). 0 disables downsampling."""

    render_goal_pass: bool = False
    """If True, run a second Madrona render pass per step with the robot posed at
    an idealized joint configuration (see goal_qpos_source) and all non-robot
    geometry pushed below the camera. Output goes to data.output["goal_rgb"].
    Doubles render cost when enabled."""

    goal_robot_asset_name: str = "robot"
    """Which articulation in scene.articulations provides the idealized chain for
    the goal pass. Only consulted when render_goal_pass=True."""

    goal_geom_sentinel_z: float = -1e6
    """World-space z value used to push non-robot geom slots out of camera frame
    during the goal pass. Should be far below any valid scene geometry."""

    goal_qpos_source: str = "raw_action"
    """Which joint-position signal the goal pass renders.

    - "target": robot.data.joint_pos_target — post-EMA, post-clamp actuator
      setpoint. Invariant to actuator dynamics ("ideal actuator" interpretation),
      but bakes in any action-term smoothing.
    - "raw_action": apply the action term's linear scale to its raw_actions
      buffer, skipping EMA / clamp. Invariant to BOTH actuator dynamics and
      action-term smoothing — shows the policy's per-step commanded pose.
      Use this when you plan to swap in a noisy/poorly-behaved actuator and
      want the goal image to track the policy command, not the actuator state.

    Read once per step at observation time, well after process_actions has
    stamped both buffers, so neither source lags step N's policy output."""

    goal_action_term_name: str = "joint_pos"
    """Name under which the (EMA) joint-position action term is registered in
    env.action_manager. Only consulted when goal_qpos_source == "raw_action"."""

    goal_camera_indices: tuple[int, ...] | None = None
    """Which camera indices emit data.output[\"goal_rgb\"] / \"goal_rgb_<i>\".

    None (default) = all cameras get a goal output. Set to e.g. (0,) to skip
    buffer allocation + per-step copy for the second camera in a multi-cam
    sensor. Madrona renders all cams in one batched call regardless, so this
    saves obs payload + encoder forward, NOT GPU render time."""

    renderer_type: str = "madrona"
    """ Needed for feature parity with Newton. Defaults to MISSING elsewhere """


@configclass
class CameraOffsetCfg:
    """Configuration for a single camera offset."""

    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset from the parent frame (x, y, z)."""

    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Orientation offset from the parent frame as quaternion (w, x, y, z)."""

    convention: str = "world"
    """Convention for the orientation. Options: "world", "opengl", "ros"."""

    body_name: str | None = None
    """Name of the articulation body this camera follows. If set, `pos`/`rot` are interpreted
    as a link-local offset and the world camera pose is recomputed every frame from
    `scene[asset_name].data.body_pos_w` and `body_quat_w`. If None, the camera is static
    in the env-local world frame (current behavior, fully back-compat)."""

    asset_name: str = "robot"
    """Scene articulation key used to resolve `body_name`. Defaults to "robot"."""


@configclass
class MadronaCameraCfg:
    """Configuration for one camera view inside a Madrona multicam sensor."""

    offset: CameraOffsetCfg = CameraOffsetCfg()
    """Pose offset for this camera view."""

    spawn: PinholeCameraCfg = MISSING
    """Pinhole camera settings for this view."""


@configclass
class MadronaMultiCamTiledCameraCfg(CameraCfg):
    """Configuration for Madrona-backed multi-camera sensor.

    This camera uses GPU rasterization via Madrona with support for multiple
    camera views rendered in a single batched call. All cameras share the same
    geometry and texture memory for maximum efficiency.

    The output dictionary will contain:
        - "rgb": First camera RGB (num_envs, H, W, 3)
        - "rgb_1": Second camera RGB (num_envs, H, W, 3)
        - "rgb_N": Nth camera RGB
        - Similar for "rgba", "depth"

    This allows using standard observation functions with different data_type params.
    """

    from .camera import MadronaTiledCamera

    class_type: type = MadronaTiledCamera  # Uses unified class

    data_types: list[str] = MISSING
    """Base data types to capture. Supported: ["rgb", "rgba", "depth"].

    Output will include these for each camera (e.g., "rgb", "rgb_1", etc.)
    """

    spawn: PinholeCameraCfg | None = None
    """Unused for multicam; each entry in :attr:`cameras` carries its own spawn config."""

    cameras: list[MadronaCameraCfg] = field(default_factory=list)
    """List of per-view camera configurations. Must have at least 2 cameras."""

    use_rasterizer: bool = True
    """Use rasterizer (True) or raytracer (False)."""

    add_debug_geometry: bool = False
    """Add debug geometry for cameras in the scene."""

    max_texture_size: int | None = None
    """Maximum texture edge (pixels) for every texture this renderer ingests.

    All camera views share one geometry/texture pool, so this applies to the
    whole multicam sensor. None falls back to the legacy
    MADRONA_MAX_TEXTURE_SIZE env var, then to the default (512). 0 disables
    downsampling."""

    renderer_type: str = "madrona"
    """ Needed for feature parity with Newton. Defaults to MISSING elsewhere """

    # Goal-image render pass — same fields as MadronaTiledCameraCfg. See that
    # class for full docs. Mirrored here because both configclasses are
    # siblings of CameraCfg and the camera implementation uses
    # getattr(self.cfg, ...) so any subclass that exposes these fields
    # participates in the goal pass.
    render_goal_pass: bool = False
    goal_robot_asset_name: str = "robot"
    goal_geom_sentinel_z: float = -1e6
    goal_qpos_source: str = "raw_action"
    goal_action_term_name: str = "joint_pos"
    goal_camera_indices: tuple[int, ...] | None = None

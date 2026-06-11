"""USD-native kinematic chain for Isaac Lab articulations.

Extracts joint axes, types, and local mount transforms by walking the
articulation's USD subtree. Reads parent body indices and DOF name->index
mapping from PhysX (`articulation.root_physx_view.shared_metatype`). Outputs
batched torch FK that takes (num_envs, num_dofs) joint positions and returns
world-frame body positions + quaternions, anchored at body 0's world pose.

No URDF required. Works for any USD-loaded PhysX articulation.

Supported joint types: revolute, prismatic, fixed. Spherical / D6 are skipped
(child body falls back to identity-relative-to-parent if no joint is found,
which is a recognizable failure mode in the smoke test).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch

# pxr / omni imports are deferred into _walk_usd to keep the module importable
# for tooling outside Isaac Sim. If the caller actually constructs a chain,
# Isaac Sim must be running.


_AXIS_MAP: dict[str, tuple[float, float, float]] = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


@dataclass
class _JointEntry:
    parent_body_idx: int
    child_body_idx: int
    dof_idx: int  # -1 for fixed
    joint_type: str  # "revolute" / "prismatic" / "fixed"
    axis_xyz: tuple[float, float, float]
    local_pos0: tuple[float, float, float]
    local_rot0_wxyz: tuple[float, float, float, float]
    local_pos1: tuple[float, float, float]
    local_rot1_wxyz: tuple[float, float, float, float]


class USDArticulationChain:
    """Batched torch FK for an Isaac Lab Articulation, extracted from USD."""

    def __init__(self, articulation, device=None, dtype: torch.dtype = torch.float32):
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else articulation.device

        view = articulation.root_physx_view
        meta = view.shared_metatype

        self.body_names: list[str] = list(meta.link_names)
        self.dof_names: list[str] = list(meta.dof_names)
        self.num_bodies: int = len(self.body_names)
        self.num_dofs: int = len(self.dof_names)

        # parent body index per body (root = -1)
        link_parent_indices = meta.link_parent_indices
        if isinstance(link_parent_indices, dict):
            parent_list = [link_parent_indices.get(name, -1) for name in self.body_names]
        else:
            parent_list = list(link_parent_indices)
        self.parent_body_idx = torch.tensor(parent_list, dtype=torch.long, device=self.device)

        # Walk USD for joint metadata
        joints_by_child = self._walk_usd(view, meta)
        self._missing_joint_bodies = [
            i for i, p in enumerate(parent_list)
            if p >= 0 and i not in joints_by_child
        ]

        # Per-body tensors aligned with body_names ordering
        self.body_dof_idx = torch.full((self.num_bodies,), -1, dtype=torch.long, device=self.device)
        self.body_joint_kind = torch.zeros((self.num_bodies,), dtype=torch.long, device=self.device)
        # 0 = fixed/unknown, 1 = revolute, 2 = prismatic
        self.body_axis = torch.zeros((self.num_bodies, 3), dtype=dtype, device=self.device)
        self.body_T0 = torch.eye(4, dtype=dtype, device=self.device).unsqueeze(0).repeat(self.num_bodies, 1, 1)
        self.body_T1_inv = torch.eye(4, dtype=dtype, device=self.device).unsqueeze(0).repeat(self.num_bodies, 1, 1)

        for child_idx, je in joints_by_child.items():
            self.body_dof_idx[child_idx] = je.dof_idx
            if je.joint_type == "revolute":
                self.body_joint_kind[child_idx] = 1
            elif je.joint_type == "prismatic":
                self.body_joint_kind[child_idx] = 2
            else:
                self.body_joint_kind[child_idx] = 0
            self.body_axis[child_idx] = torch.tensor(je.axis_xyz, dtype=dtype, device=self.device)
            self.body_T0[child_idx] = self._make_mat4(je.local_pos0, je.local_rot0_wxyz)
            T1 = self._make_mat4(je.local_pos1, je.local_rot1_wxyz)
            self.body_T1_inv[child_idx] = self._invert_rigid(T1)

        # Topological ordering: process parents before children
        self._topo_order = self._compute_topological_order(parent_list)

    # ------------------------------------------------------------------
    # Public FK
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        joint_pos: torch.Tensor,
        root_pose_w: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute world body poses.

        Args:
            joint_pos: (num_envs, num_dofs) joint values in DOF order
                       (matches `articulation.data.joint_pos`).
            root_pose_w: (num_envs, 7) [x,y,z, qw,qx,qy,qz] world pose of body 0
                in Isaac Lab convention. Matches
                `articulation.data.body_link_pose_w[:, 0]`.
                If None, body 0 is at origin (chain-relative output).

        Returns:
            body_pos: (num_envs, num_bodies, 3) world position
            body_quat_wxyz: (num_envs, num_bodies, 4) world orientation as wxyz
                            (matches `articulation.data.body_link_pose_w[:, :, 3:7]`)
        """
        device, dtype = self.device, self.dtype
        joint_pos = joint_pos.to(device=device, dtype=dtype)
        E = joint_pos.shape[0]
        B = self.num_bodies

        # body_world: (E, B, 4, 4), initialized to identity
        body_world = torch.eye(4, device=device, dtype=dtype).expand(E, B, 4, 4).contiguous()

        if root_pose_w is not None:
            root_pose_w = root_pose_w.to(device=device, dtype=dtype)
            body_world[:, 0] = self._pose7_wxyz_to_mat4(root_pose_w)

        for body_idx in self._topo_order:
            parent_idx = int(self.parent_body_idx[body_idx].item())
            if parent_idx < 0:
                continue  # root, already set

            T0 = self.body_T0[body_idx]            # (4, 4)
            T1_inv = self.body_T1_inv[body_idx]    # (4, 4)
            kind = int(self.body_joint_kind[body_idx].item())
            dof_idx = int(self.body_dof_idx[body_idx].item())

            if kind == 1 and dof_idx >= 0:        # revolute
                q = joint_pos[:, dof_idx]
                joint_motion = self._axis_angle_to_mat4(self.body_axis[body_idx], q)  # (E, 4, 4)
            elif kind == 2 and dof_idx >= 0:      # prismatic
                q = joint_pos[:, dof_idx]
                joint_motion = self._translate_along_axis_mat4(self.body_axis[body_idx], q)
            else:                                  # fixed or missing
                joint_motion = torch.eye(4, device=device, dtype=dtype).expand(E, 4, 4)

            parent_world = body_world[:, parent_idx]                     # (E, 4, 4)
            child_world = parent_world @ T0 @ joint_motion @ T1_inv      # (E, 4, 4)
            body_world[:, body_idx] = child_world

        body_pos = body_world[..., :3, 3]
        body_quat_wxyz = self._rotmat_to_quat_wxyz(body_world[..., :3, :3])
        return body_pos, body_quat_wxyz

    # ------------------------------------------------------------------
    # USD walker
    # ------------------------------------------------------------------

    def _walk_usd(self, view, meta) -> dict[int, _JointEntry]:
        import omni.usd
        from pxr import Gf, Usd, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        root_path = view.prim_paths[0]  # env 0; topology identical across envs
        root_prim = stage.GetPrimAtPath(root_path)
        if not root_prim or not root_prim.IsValid():
            raise RuntimeError(f"USDArticulationChain: invalid root prim at '{root_path}'")

        link_indices: dict[str, int] = (
            dict(meta.link_indices)
            if hasattr(meta, "link_indices")
            else {n: i for i, n in enumerate(meta.link_names)}
        )
        dof_indices: dict[str, int] = (
            dict(meta.dof_indices)
            if hasattr(meta, "dof_indices")
            else {n: i for i, n in enumerate(meta.dof_names)}
        )

        joints_by_child: dict[int, _JointEntry] = {}
        for prim in Usd.PrimRange(root_prim):
            joint_prim = UsdPhysics.Joint(prim)
            if not joint_prim:
                continue
            is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
            is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
            is_fixed = prim.IsA(UsdPhysics.FixedJoint)
            if not (is_revolute or is_prismatic or is_fixed):
                continue

            body1_targets = joint_prim.GetBody1Rel().GetTargets()
            if not body1_targets:
                continue
            child_name = body1_targets[0].name
            child_idx = link_indices.get(child_name)
            if child_idx is None:
                continue

            body0_targets = joint_prim.GetBody0Rel().GetTargets()
            parent_idx = (
                link_indices.get(body0_targets[0].name, -1)
                if body0_targets else -1
            )

            joint_name = prim.GetName()
            if is_revolute:
                joint_type = "revolute"
                axis_str = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() or "X"
                dof_idx = dof_indices.get(joint_name, -1)
            elif is_prismatic:
                joint_type = "prismatic"
                axis_str = UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get() or "X"
                dof_idx = dof_indices.get(joint_name, -1)
            else:
                joint_type = "fixed"
                axis_str = "X"
                dof_idx = -1

            local_pos0 = joint_prim.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
            local_rot0 = joint_prim.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
            local_pos1 = joint_prim.GetLocalPos1Attr().Get() or Gf.Vec3f(0, 0, 0)
            local_rot1 = joint_prim.GetLocalRot1Attr().Get() or Gf.Quatf(1, 0, 0, 0)

            joints_by_child[child_idx] = _JointEntry(
                parent_body_idx=parent_idx,
                child_body_idx=child_idx,
                dof_idx=dof_idx,
                joint_type=joint_type,
                axis_xyz=_AXIS_MAP.get(axis_str, (1.0, 0.0, 0.0)),
                local_pos0=tuple(float(v) for v in local_pos0),
                local_rot0_wxyz=(
                    float(local_rot0.GetReal()),
                    *(float(v) for v in local_rot0.GetImaginary()),
                ),
                local_pos1=tuple(float(v) for v in local_pos1),
                local_rot1_wxyz=(
                    float(local_rot1.GetReal()),
                    *(float(v) for v in local_rot1.GetImaginary()),
                ),
            )
        return joints_by_child

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_topological_order(parent_list: list[int]) -> list[int]:
        n = len(parent_list)
        children: list[list[int]] = [[] for _ in range(n)]
        roots: list[int] = []
        for i, p in enumerate(parent_list):
            if p < 0:
                roots.append(i)
            else:
                children[p].append(i)
        order: list[int] = []
        visited = [False] * n
        queue = deque(roots)
        for r in roots:
            visited[r] = True
        while queue:
            i = queue.popleft()
            order.append(i)
            for c in children[i]:
                if not visited[c]:
                    visited[c] = True
                    queue.append(c)
        # any disconnected bodies (shouldn't happen for a proper articulation)
        # are appended at the end so we still touch every body
        for i in range(n):
            if not visited[i]:
                order.append(i)
        return order

    def _make_mat4(
        self,
        pos_xyz: tuple[float, float, float],
        quat_wxyz: tuple[float, float, float, float],
    ) -> torch.Tensor:
        device, dtype = self.device, self.dtype
        M = torch.eye(4, device=device, dtype=dtype)
        q = torch.tensor(quat_wxyz, device=device, dtype=dtype)
        M[:3, :3] = self._quat_wxyz_to_rotmat(q)
        M[:3, 3] = torch.tensor(pos_xyz, device=device, dtype=dtype)
        return M

    @staticmethod
    def _invert_rigid(T: torch.Tensor) -> torch.Tensor:
        Tinv = torch.eye(4, device=T.device, dtype=T.dtype)
        R_T = T[:3, :3].T
        Tinv[:3, :3] = R_T
        Tinv[:3, 3] = -R_T @ T[:3, 3]
        return Tinv

    @staticmethod
    def _quat_wxyz_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        qw, qx, qy, qz = q.unbind(-1)
        return torch.stack([
            1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),     2 * (qx * qz + qw * qy),
            2 * (qx * qy + qw * qz),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
            2 * (qx * qz - qw * qy),     2 * (qy * qz + qw * qx),     1 - 2 * (qx * qx + qy * qy),
        ], dim=-1).reshape(*q.shape[:-1], 3, 3)

    @staticmethod
    def _pose7_wxyz_to_mat4(pose: torch.Tensor) -> torch.Tensor:
        # pose: (..., 7) [x,y,z, qw,qx,qy,qz] (Isaac Lab convention) -> (..., 4, 4)
        q_wxyz = pose[..., 3:7]
        R = USDArticulationChain._quat_wxyz_to_rotmat(q_wxyz)
        M = torch.zeros(*pose.shape[:-1], 4, 4, device=pose.device, dtype=pose.dtype)
        M[..., :3, :3] = R
        M[..., :3, 3] = pose[..., :3]
        M[..., 3, 3] = 1.0
        return M

    @staticmethod
    def _axis_angle_to_mat4(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        # axis: (3,), angle: (E,) -> (E, 4, 4). axis assumed unit-length.
        E = angle.shape[0]
        device, dtype = angle.device, angle.dtype
        c = torch.cos(angle)
        s = torch.sin(angle)
        omc = 1.0 - c
        ax, ay, az = axis[0], axis[1], axis[2]
        R = torch.stack([
            c + ax * ax * omc,      ax * ay * omc - az * s, ax * az * omc + ay * s,
            ay * ax * omc + az * s, c + ay * ay * omc,      ay * az * omc - ax * s,
            az * ax * omc - ay * s, az * ay * omc + ax * s, c + az * az * omc,
        ], dim=-1).reshape(E, 3, 3)
        M = torch.zeros(E, 4, 4, device=device, dtype=dtype)
        M[:, :3, :3] = R
        M[:, 3, 3] = 1.0
        return M

    @staticmethod
    def _translate_along_axis_mat4(axis: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
        E = dist.shape[0]
        device, dtype = dist.device, dist.dtype
        M = torch.eye(4, device=device, dtype=dtype).expand(E, 4, 4).contiguous()
        M[:, :3, 3] = dist.unsqueeze(-1) * axis.unsqueeze(0)
        return M

    @staticmethod
    def _rotmat_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
        # Shepperd's method, batched & branchless via torch.where. Returns wxyz.
        m = R.reshape(*R.shape[:-2], 9)
        m00, m01, m02 = m[..., 0], m[..., 1], m[..., 2]
        m10, m11, m12 = m[..., 3], m[..., 4], m[..., 5]
        m20, m21, m22 = m[..., 6], m[..., 7], m[..., 8]
        trace = m00 + m11 + m22

        eps = 1e-20
        SA = torch.sqrt(torch.clamp(trace + 1.0, min=eps)) * 2.0
        qwA = 0.25 * SA
        qxA = (m21 - m12) / SA
        qyA = (m02 - m20) / SA
        qzA = (m10 - m01) / SA
        SB = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=eps)) * 2.0
        qwB = (m21 - m12) / SB
        qxB = 0.25 * SB
        qyB = (m01 + m10) / SB
        qzB = (m02 + m20) / SB
        SC = torch.sqrt(torch.clamp(1.0 + m11 - m00 - m22, min=eps)) * 2.0
        qwC = (m02 - m20) / SC
        qxC = (m01 + m10) / SC
        qyC = 0.25 * SC
        qzC = (m12 + m21) / SC
        SD = torch.sqrt(torch.clamp(1.0 + m22 - m00 - m11, min=eps)) * 2.0
        qwD = (m10 - m01) / SD
        qxD = (m02 + m20) / SD
        qyD = (m12 + m21) / SD
        qzD = 0.25 * SD

        condA = trace > 0
        condB = (~condA) & (m00 > m11) & (m00 > m22)
        condC = (~condA) & (~condB) & (m11 > m22)

        qw = torch.where(condA, qwA, torch.where(condB, qwB, torch.where(condC, qwC, qwD)))
        qx = torch.where(condA, qxA, torch.where(condB, qxB, torch.where(condC, qxC, qxD)))
        qy = torch.where(condA, qyA, torch.where(condB, qyB, torch.where(condC, qyC, qyD)))
        qz = torch.where(condA, qzA, torch.where(condB, qzB, torch.where(condC, qzC, qzD)))
        return torch.stack([qw, qx, qy, qz], dim=-1)

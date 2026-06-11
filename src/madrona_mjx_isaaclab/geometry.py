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

"""Geometry extraction and conversion for Madrona renderer.

VERSION: 2025-12-03-v2 (texture debug)

Converts USD geometry to Madrona's expected format (vertices, normals, indices).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TextureData:
    """Texture data for Madrona."""

    pixels: np.ndarray  # (H, W, C) uint8
    width: int
    height: int
    channels: int


@dataclass
class MadronaGeometry:
    """Consolidated geometry data for Madrona initialization."""

    vertices: np.ndarray  # (total_verts, 3) float32
    normals: np.ndarray  # (total_verts, 3) float32
    indices: np.ndarray  # (total_tris, 3) int32
    uvs: np.ndarray  # (total_verts, 2) float32 - texture coordinates
    colors: np.ndarray  # (num_geoms, 4) float32 RGBA per geometry
    geom_vertex_offsets: list[int]  # start vertex index for each geom
    geom_index_offsets: list[int]  # start triangle index for each geom
    geom_uv_offsets: list[int]  # start UV index for each geom
    textures: list[TextureData]  # list of textures
    geom_texture_ids: list[int]  # texture index per geom (-1 = no texture)


@dataclass
class SingleGeometry:
    """Geometry data for a single mesh/primitive."""

    vertices: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    indices: np.ndarray  # (M, 3) int32
    uvs: np.ndarray | None  # (N, 2) float32 or None
    color: np.ndarray  # (4,) float32 RGBA
    texture: TextureData | None  # texture if available


class GeometryAdapter(ABC):
    """Base class for adapting different USD prim types to Madrona geometry."""

    @abstractmethod
    def extract(self, prim) -> SingleGeometry | None:
        """Extract geometry data from USD prim."""
        pass


DEFAULT_MAX_TEXTURE_SIZE = 512


def resolve_max_texture_size(configured: int | None = None) -> int:
    """Resolve the texture max-edge for a renderer instance.

    Precedence: explicit camera-config value, then the legacy
    MADRONA_MAX_TEXTURE_SIZE env var, then DEFAULT_MAX_TEXTURE_SIZE.
    A value of 0 disables downsampling.
    """
    if configured is not None:
        return int(configured)
    import os

    raw = os.environ.get("MADRONA_MAX_TEXTURE_SIZE")
    if raw is not None:
        return int(raw)
    return DEFAULT_MAX_TEXTURE_SIZE


class MeshGeometryAdapter(GeometryAdapter):
    """Adapter for UsdGeom.Mesh prims."""

    # Tracks already-logged texture paths to avoid duplicate log messages
    _logged_textures: set[str] = set()

    def __init__(self, spec=None, max_texture_size: int | None = None):
        self._spec = spec
        self._max_texture_size = max_texture_size

    @staticmethod
    def _repo_root() -> Path:
        """Return the workspace root for local asset fallbacks."""
        for parent in Path(__file__).resolve().parents:
            if (parent / ".git").exists():
                return parent
        return Path.cwd()

    def extract(self, prim) -> SingleGeometry | None:
        from pxr import Gf, UsdGeom

        mesh = UsdGeom.Mesh(prim)

        points_attr = mesh.GetPointsAttr()
        if not points_attr.HasValue():
            return None

        points = np.array(points_attr.Get(), dtype=np.float32)

        # Apply FULL local-to-parent transform to get correctly positioned vertices
        # This is critical: mesh vertices need to be in body-local coordinates,
        # but USD mesh prims may have their own local transform relative to parent
        xformable = UsdGeom.Xformable(prim)
        local_xform = xformable.GetLocalTransformation()

        # Extract translation
        translation = local_xform.ExtractTranslation()
        trans_np = np.array(
            [translation[0], translation[1], translation[2]], dtype=np.float32
        )

        # Extract rotation as 3x3 matrix
        rot_mat = Gf.Matrix3d(local_xform.ExtractRotationMatrix())

        # Extract scale
        scale = np.array(
            [
                abs(local_xform.GetRow(0).GetLength()),
                abs(local_xform.GetRow(1).GetLength()),
                abs(local_xform.GetRow(2).GetLength()),
            ],
            dtype=np.float32,
        )

        # Apply full transform: scale, rotate, then translate
        # USD separates scale from rotation, so we can apply them independently

        # First scale
        points = points * scale

        # Then rotate (ExtractRotationMatrix already gives us scale-free rotation)
        rot_np = np.array(
            [[rot_mat[i, j] for j in range(3)] for i in range(3)], dtype=np.float32
        )
        points = points @ rot_np.T

        # Finally translate
        points = points + trans_np

        face_counts_attr = mesh.GetFaceVertexCountsAttr()
        face_indices_attr = mesh.GetFaceVertexIndicesAttr()

        if not face_counts_attr.HasValue() or not face_indices_attr.HasValue():
            return None

        face_vertex_counts = np.array(face_counts_attr.Get())
        face_vertex_indices = np.array(face_indices_attr.Get())

        # Check for face-varying UVs BEFORE triangulation
        # If face-varying, we need to expand vertices to match UV layout
        uvs, is_face_varying = self._extract_uvs_with_interp(
            mesh, len(points), face_vertex_indices
        )

        if is_face_varying and uvs is not None:
            from madrona_mjx_isaaclab.mesh_decimation.core import expand_face_varying
            points, uvs, indices = expand_face_varying(
                points, face_vertex_counts, face_vertex_indices, uvs
            )
        else:
            indices = self._triangulate(face_vertex_counts, face_vertex_indices)

        # Decimate mesh using shared simplify_mesh (handles UV-bearing meshes via simplifyWithAttributes)
        if self._spec is not None:
            from madrona_mjx_isaaclab.mesh_decimation.core import simplify_mesh
            points, indices, uvs = simplify_mesh(points, indices, self._spec, uvs=uvs)

        # Compute normals after decimation
        normals_attr = mesh.GetNormalsAttr()
        if normals_attr.HasValue() and len(np.array(normals_attr.Get())) == len(points):
            normals = np.array(normals_attr.Get(), dtype=np.float32)
        else:
            normals = self._compute_normals(points, indices)

        # Extract color and texture
        color = self._extract_color(prim)
        texture = self._extract_texture(prim)

        return SingleGeometry(
            vertices=points,
            normals=normals,
            indices=indices,
            uvs=uvs,
            color=color,
            texture=texture,
        )

    def _extract_uvs_with_interp(
        self, mesh, num_verts: int, face_indices: np.ndarray = None
    ) -> tuple[np.ndarray | None, bool]:
        """Extract UV coordinates from mesh primvars with interpolation info.

        Returns:
            tuple of (uvs, is_face_varying)
            - uvs: UV coordinates array or None
            - is_face_varying: True if UVs are face-varying (per face-vertex)
        """
        from pxr import UsdGeom

        # Try common UV primvar names
        for uv_name in ["st", "UVMap", "uv"]:
            primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).GetPrimvar(uv_name)
            if primvar and primvar.HasValue():
                uvs = np.array(primvar.Get(), dtype=np.float32)
                interp = primvar.GetInterpolation()

                if len(uvs) == num_verts:
                    # Vertex-varying - direct match
                    return uvs, False
                elif interp == "faceVarying" and face_indices is not None:
                    # Face-varying UVs - one UV per face vertex
                    # This is common in USD meshes where each face corner has its own UV
                    if len(uvs) == len(face_indices):
                        # UVs are per face-vertex, need vertex expansion
                        return uvs, True

        return None, False

    def _extract_uvs(
        self, mesh, num_verts: int, face_indices: np.ndarray = None
    ) -> np.ndarray | None:
        """Extract UV coordinates from mesh primvars (legacy wrapper).

        Note: This is a legacy wrapper. New code should use _extract_uvs_with_interp.
        """
        uvs, _ = self._extract_uvs_with_interp(mesh, num_verts, face_indices)
        return uvs

    @staticmethod
    def _resolve_surface_shader(material) -> tuple:
        """Return (shader, tag) where tag is 'mdl', 'preview', or 'unknown'.

        IMPORTANT: USD's `ComputeSurfaceSource("mdl")` will fall back and return
        a UsdPreviewSurface shader if no MDL surface terminal exists. So we
        cannot trust the rendercontext alone -- always check `info:id` to
        distinguish a true MDL shader from a UsdPreviewSurface that was returned
        as a fallback. This was the root cause of `--color_all` not pinking the
        robot in Madrona: the override material is a UsdPreviewSurface, but our
        resolver tagged it as MDL and then `_extract_color` looked for the MDL
        `diffuse_color_constant` input (which doesn't exist), falling back to
        the default gray.
        """
        from pxr import UsdShade

        def _classify(shader) -> str:
            if shader is None or not shader.GetPrim().IsValid():
                return "unknown"
            id_attr = shader.GetPrim().GetAttribute("info:id")
            shader_id = id_attr.Get() if id_attr else None
            if shader_id == "UsdPreviewSurface":
                return "preview"
            # MDL surfaces typically have info:mdl:sourceAsset set; the
            # shader_id may be empty or a vendor token. Treat as MDL only when
            # we actually see an MDL asset reference.
            mdl_attr = shader.GetPrim().GetAttribute("info:mdl:sourceAsset")
            if mdl_attr and mdl_attr.Get():
                return "mdl"
            # Some MDL exports use the shader_id as the MDL function name
            # (e.g. "OmniPBR"). Accept those too.
            if shader_id and shader_id not in ("UsdPreviewSurface",):
                return "mdl"
            return "unknown"

        for render_context in ("mdl", "universal", ""):
            shader_info = material.ComputeSurfaceSource(render_context)
            if shader_info and shader_info[0] and shader_info[0].GetPrim().IsValid():
                tag = _classify(shader_info[0])
                if tag != "unknown":
                    return (shader_info[0], tag)

        mat_prim = material.GetPrim()
        for child in mat_prim.GetChildren():
            child_shader = UsdShade.Shader(child)
            if child_shader and child_shader.GetPrim().IsValid():
                tag = _classify(child_shader)
                return (child_shader, tag if tag != "unknown" else "unknown")

        return (None, "unknown")

    def _extract_texture(self, prim) -> TextureData | None:
        """Extract texture from material binding using shader-graph aware logic."""
        from pxr import Sdf, UsdShade

        binding_api = UsdShade.MaterialBindingAPI(prim)
        result = binding_api.ComputeBoundMaterial()
        material = result[0] if result else None

        if not material:
            return None

        shader, tag = self._resolve_surface_shader(material)

        if tag == "mdl" and shader is not None:
            inp = shader.GetInput("diffuse_texture")
            if inp:
                value = inp.Get()
                if isinstance(value, Sdf.AssetPath):
                    return self._load_texture(value, shader.GetPrim())
            return None

        if tag == "preview" and shader is not None:
            diffuse_inp = shader.GetInput("diffuseColor")
            if diffuse_inp:
                # GetConnectedSources() returns (List[ConnectionSourceInfo], List[Sdf.Path]);
                # the outer tuple is always truthy, so check the inner connection list.
                connections, _invalid = diffuse_inp.GetConnectedSources()
                if connections:
                    info = connections[0]
                    upstream_shader = UsdShade.Shader(info.source.GetPrim())
                    file_inp = upstream_shader.GetInput("file")
                    if file_inp:
                        value = file_inp.Get()
                        if isinstance(value, Sdf.AssetPath):
                            return self._load_texture(value, upstream_shader.GetPrim())
            return None

        mat_prim = material.GetPrim()
        for child in mat_prim.GetChildren():
            shader_prim = UsdShade.Shader(child)
            if not shader_prim:
                continue
            for inp in shader_prim.GetInputs():
                input_name = inp.GetBaseName().lower()
                value = inp.Get()
                if (
                    "file" in input_name
                    or "diffuse" in input_name
                    or "albedo" in input_name
                ):
                    if value is not None:
                        if hasattr(value, "path"):
                            texture_path = str(value.path)
                            return self._load_texture(texture_path, child.GetPrim())
                        elif isinstance(value, str):
                            return self._load_texture(value, child.GetPrim())

        return None

    def _load_texture(
        self, path, prim=None, max_size: int | None = None
    ) -> TextureData | None:
        """Load texture from file path with optional downsampling.

        Args:
            path: Texture path string, or Sdf.AssetPath. Empty AssetPath returns None silently.
            prim: USD prim to resolve relative paths from (optional)
            max_size: Maximum texture dimension. If None, uses the adapter's
                      max_texture_size (set per camera config via the factory).
                      Set to 0 to disable downsampling.
        """
        import io
        import os
        from urllib.parse import urlparse

        from pxr import Sdf

        if isinstance(path, Sdf.AssetPath):
            asset_path = path
            resolved = asset_path.resolvedPath
            raw = asset_path.path
            if not raw or raw in ("@", "@@"):
                return None
            if resolved and os.path.exists(resolved):
                path = resolved
            else:
                path = raw

        if max_size is None:
            max_size = resolve_max_texture_size(self._max_texture_size)

        # Handle asset path markers
        if path.startswith("@"):
            path = path[1:]
        if path.endswith("@"):
            path = path[:-1]

        if not path:
            return None

        def _is_remote(asset_path: str) -> bool:
            parsed = urlparse(asset_path)
            return parsed.scheme in {"http", "https", "omniverse"}

        def _is_absolute(asset_path: str) -> bool:
            return os.path.isabs(asset_path) or _is_remote(asset_path)

        def _join_asset_path(base_path: str, rel_path: str) -> str:
            """Join a relative asset path against a local or remote layer path."""
            rel_path = rel_path.replace("\\", "/")
            if _is_remote(base_path):
                prefix, rest = base_path.split("://", 1)
                asset_dir = rest.rsplit("/", 1)[0] if "/" in rest else rest
                return prefix + "://" + os.path.normpath(
                    os.path.join(asset_dir, rel_path)
                ).replace("\\", "/")
            return os.path.normpath(os.path.join(os.path.dirname(base_path), rel_path))

        local_candidates: list[str] = []

        def _add_local_candidate(candidate: str | os.PathLike[str] | None):
            if candidate is None:
                return
            candidate_str = os.fspath(candidate)
            if candidate_str not in local_candidates:
                local_candidates.append(candidate_str)

        def _add_repo_asset_candidates(asset_path: str):
            repo_root = self._repo_root()
            asset_name = os.path.basename(asset_path)
            if not asset_name:
                return
            _add_local_candidate(repo_root / "assets" / "Materials" / asset_name)
            _add_local_candidate(
                repo_root
                / "assets"
                / "Props"
                / "Blocks"
                / "DexCube"
                / "Materials"
                / asset_name
            )
            _add_local_candidate(
                repo_root
                / "assets"
                / "Props"
                / "DexCube"
                / "Materials"
                / asset_name
            )
            _add_local_candidate(
                repo_root / "assets" / "Props" / "Materials" / asset_name
            )
            if not _is_absolute(asset_path):
                _add_local_candidate(Path.cwd() / asset_path)
                _add_local_candidate(repo_root / "assets" / asset_path)

        # First, try to resolve the path using USD's asset resolution system
        # This handles relative paths, Nucleus URLs, and search paths properly
        resolved_path = path
        try:
            from pxr import Ar

            resolver = Ar.GetResolver()
            resolved = resolver.Resolve(path)
            if resolved:
                resolved_path = (
                    str(resolved.GetPathString())
                    if hasattr(resolved, "GetPathString")
                    else str(resolved)
                )
        except Exception as e:
            print(f"[MeshGeometryAdapter] USD resolver failed: {e}")

        # If still relative, try resolving relative to the prim's defining layer
        if not _is_absolute(resolved_path):
            try:
                from pxr import Sdf

                # First, try to get the layer that defines the prim (where the texture ref lives)
                anchor_layer = None
                if prim is not None:
                    # Get the layer stack and find the layer that introduced this prim.
                    prim_stack = prim.GetPrimStack()
                    if prim_stack:
                        # The first spec in the stack is from the strongest layer.
                        anchor_layer = prim_stack[0].layer

                # Fallback to stage root layer
                if anchor_layer is None and prim is not None:
                    stage = prim.GetStage()
                    if stage:
                        anchor_layer = stage.GetRootLayer()

                if anchor_layer:
                    # Use Sdf.ComputeAssetPathRelativeToLayer for proper resolution
                    abs_path = Sdf.ComputeAssetPathRelativeToLayer(anchor_layer, path)
                    if abs_path:
                        resolved_path = abs_path
                    else:
                        # Fallback: manual resolution using layer identifier
                        layer_path = anchor_layer.realPath or anchor_layer.identifier
                        if layer_path:
                            resolved_path = _join_asset_path(layer_path, path)
            except Exception as e:
                print(f"[MeshGeometryAdapter] Layer-relative resolution failed: {e}")

        if not _is_remote(resolved_path):
            _add_local_candidate(resolved_path)
        _add_repo_asset_candidates(path)
        _add_repo_asset_candidates(resolved_path)

        # Try using Omniverse client to read the texture (works for Nucleus URLs and local files)
        texture_data = None
        loaded_from = resolved_path
        try:
            import omni.client

            result, _, content = omni.client.read_file(resolved_path)
            if result == omni.client.Result.OK:
                texture_data = bytes(content)
        except Exception as e:
            print(
                f"[MeshGeometryAdapter] omni.client.read_file failed for {resolved_path}: {e}"
            )

        # If omni.client failed, try direct local file reads.
        if texture_data is None:
            for candidate in local_candidates:
                if (
                    not candidate
                    or _is_remote(candidate)
                    or not os.path.exists(candidate)
                ):
                    continue
                try:
                    with open(candidate, "rb") as f:
                        texture_data = f.read()
                    loaded_from = candidate
                    break
                except Exception as e:
                    print(
                        f"[MeshGeometryAdapter] Failed to read local file {candidate}: {e}"
                    )

        if texture_data is None:
            print(
                f"[MeshGeometryAdapter] Could not load texture: {path} "
                f"(resolved={resolved_path})"
            )
            return None

        try:
            from PIL import Image

            img = Image.open(io.BytesIO(texture_data))

            if max_size > 0:
                from madrona_mjx_isaaclab.mesh_decimation.texture import downsample_texture
                img = downsample_texture(img, max_size)
                if path not in MeshGeometryAdapter._logged_textures:
                    print(
                        f"[MeshGeometryAdapter] Texture after downsample: {img.width}x{img.height} (max_size={max_size})"
                    )

            img = img.convert("RGBA")
            pixels = np.array(img, dtype=np.uint8)
            if path not in MeshGeometryAdapter._logged_textures:
                vram_kb = (pixels.nbytes) / 1024
                print(
                    f"[MeshGeometryAdapter] Loaded texture: {path} -> {loaded_from} "
                    f"({img.width}x{img.height}, {vram_kb:.1f} KB)"
                )
                MeshGeometryAdapter._logged_textures.add(path)
            return TextureData(
                pixels=pixels,
                width=img.width,
                height=img.height,
                channels=4,
            )
        except Exception as e:
            print(f"[MeshGeometryAdapter] Failed to decode texture {path}: {e}")
            return None

    def _triangulate(self, counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Convert polygon soup to triangle list via fan triangulation."""
        from madrona_mjx_isaaclab.mesh_decimation.core import triangulate
        return triangulate(counts, indices)

    def _compute_normals(self, vertices: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Compute per-vertex normals from triangle faces."""
        normals = np.zeros_like(vertices)
        for tri in indices:
            v0, v1, v2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            n = np.cross(v1 - v0, v2 - v0)
            normals[tri[0]] += n
            normals[tri[1]] += n
            normals[tri[2]] += n
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return (normals / norms).astype(np.float32)

    def _extract_color(self, prim) -> np.ndarray:
        """Extract display color from prim or material, MDL-aware."""
        from pxr import UsdGeom, UsdShade

        try:
            binding_api = UsdShade.MaterialBindingAPI(prim)
            result = binding_api.ComputeBoundMaterial()
            material = result[0] if result else None

            if material:
                shader, tag = self._resolve_surface_shader(material)

                if tag == "mdl" and shader is not None:
                    inp = shader.GetInput("diffuse_color_constant")
                    if inp:
                        value = inp.Get()
                        if value is not None and hasattr(value, "__len__") and len(value) >= 3:
                            # OmniPBR rendered diffuse is constant * tint (the
                            # "tint" name is misleading -- it is a multiplier,
                            # not a separate hue). Without this, robot links
                            # rendered as light gray when the source had tint
                            # near zero (intended near-black).
                            tint_inp = shader.GetInput("diffuse_tint")
                            tint = (1.0, 1.0, 1.0)
                            if tint_inp:
                                tv = tint_inp.Get()
                                if tv is not None and hasattr(tv, "__len__") and len(tv) >= 3:
                                    tint = (tv[0], tv[1], tv[2])
                            return np.array(
                                [value[0] * tint[0], value[1] * tint[1], value[2] * tint[2], 1.0],
                                dtype=np.float32,
                            )

                elif tag == "preview" and shader is not None:
                    inp = shader.GetInput("diffuseColor")
                    if inp:
                        connections, _invalid = inp.GetConnectedSources()
                        if not connections:
                            value = inp.Get()
                            if value is not None and hasattr(value, "__len__") and len(value) >= 3:
                                return np.array([value[0], value[1], value[2], 1.0], dtype=np.float32)

                else:
                    for child in material.GetPrim().GetChildren():
                        shader_prim = UsdShade.Shader(child)
                        if not shader_prim:
                            continue
                        for inp in shader_prim.GetInputs():
                            input_name = inp.GetBaseName().lower()
                            if "diffuse_color_constant" in input_name or "basecolor" in input_name:
                                value = inp.Get()
                                if value is not None and hasattr(value, "__len__") and len(value) >= 3:
                                    return np.array([value[0], value[1], value[2], 1.0], dtype=np.float32)
        except Exception:
            pass

        gprim = UsdGeom.Gprim(prim)
        color_attr = gprim.GetDisplayColorAttr()
        if color_attr.HasValue():
            colors = color_attr.Get()
            if colors and len(colors) > 0:
                c = colors[0]
                return np.array([c[0], c[1], c[2], 1.0], dtype=np.float32)

        return np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)


class CubeGeometryAdapter(GeometryAdapter):
    """Adapter for UsdGeom.Cube prims."""

    _VERTICES = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    _INDICES = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # front
            [4, 5, 6],
            [4, 6, 7],  # back
            [0, 1, 5],
            [0, 5, 4],  # bottom
            [2, 3, 7],
            [2, 7, 6],  # top
            [0, 4, 7],
            [0, 7, 3],  # left
            [1, 2, 6],
            [1, 6, 5],  # right
        ],
        dtype=np.int32,
    )

    _NORMALS = np.array(
        [
            [-0.577, -0.577, -0.577],
            [0.577, -0.577, -0.577],
            [0.577, 0.577, -0.577],
            [-0.577, 0.577, -0.577],
            [-0.577, -0.577, 0.577],
            [0.577, -0.577, 0.577],
            [0.577, 0.577, 0.577],
            [-0.577, 0.577, 0.577],
        ],
        dtype=np.float32,
    )

    def extract(self, prim) -> SingleGeometry | None:
        from pxr import UsdGeom

        cube = UsdGeom.Cube(prim)
        size = cube.GetSizeAttr().Get() or 1.0

        vertices = self._VERTICES * size
        color = self._extract_color(prim)

        return SingleGeometry(
            vertices=vertices.copy(),
            normals=self._NORMALS.copy(),
            indices=self._INDICES.copy(),
            uvs=None,
            color=color,
            texture=None,
        )

    def _extract_color(self, prim) -> np.ndarray:
        from pxr import UsdGeom

        gprim = UsdGeom.Gprim(prim)
        color_attr = gprim.GetDisplayColorAttr()
        if color_attr.HasValue():
            colors = color_attr.Get()
            if colors and len(colors) > 0:
                c = colors[0]
                return np.array([c[0], c[1], c[2], 1.0], dtype=np.float32)

        return np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)


class SphereGeometryAdapter(GeometryAdapter):
    """Adapter for UsdGeom.Sphere prims."""

    def __init__(self, segments: int = 16, rings: int = 8):
        self.segments = segments
        self.rings = rings

    def extract(self, prim) -> SingleGeometry | None:
        from pxr import UsdGeom

        sphere = UsdGeom.Sphere(prim)
        radius = sphere.GetRadiusAttr().Get() or 1.0

        vertices, normals, indices = self._generate_sphere(radius)
        color = self._extract_color(prim)

        return SingleGeometry(
            vertices=vertices,
            normals=normals,
            indices=indices,
            uvs=None,
            color=color,
            texture=None,
        )

    def _generate_sphere(
        self, radius: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate sphere mesh procedurally."""
        vertices = []
        normals = []

        for i in range(self.rings + 1):
            phi = np.pi * i / self.rings
            for j in range(self.segments):
                theta = 2 * np.pi * j / self.segments
                x = np.sin(phi) * np.cos(theta)
                y = np.cos(phi)
                z = np.sin(phi) * np.sin(theta)
                normals.append([x, y, z])
                vertices.append([x * radius, y * radius, z * radius])

        indices = []
        for i in range(self.rings):
            for j in range(self.segments):
                next_j = (j + 1) % self.segments
                curr = i * self.segments + j
                next_ring = (i + 1) * self.segments + j
                curr_next = i * self.segments + next_j
                next_ring_next = (i + 1) * self.segments + next_j

                if i != 0:
                    indices.append([curr, next_ring, curr_next])
                if i != self.rings - 1:
                    indices.append([curr_next, next_ring, next_ring_next])

        return (
            np.array(vertices, dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.int32),
        )

    def _extract_color(self, prim) -> np.ndarray:
        from pxr import UsdGeom

        gprim = UsdGeom.Gprim(prim)
        color_attr = gprim.GetDisplayColorAttr()
        if color_attr.HasValue():
            colors = color_attr.Get()
            if colors and len(colors) > 0:
                c = colors[0]
                return np.array([c[0], c[1], c[2], 1.0], dtype=np.float32)

        return np.array([0.8, 0.8, 0.8, 1.0], dtype=np.float32)


class GeometryAdapterFactory:
    """Factory for creating appropriate geometry adapters."""

    _adapters: dict[str, type[GeometryAdapter]] = {
        "Mesh": MeshGeometryAdapter,
        "Cube": CubeGeometryAdapter,
        "Sphere": SphereGeometryAdapter,
    }

    @classmethod
    def create(
        cls, prim_type: str, max_texture_size: int | None = None
    ) -> GeometryAdapter | None:
        adapter_class = cls._adapters.get(prim_type)
        if adapter_class is None:
            return None
        if adapter_class is MeshGeometryAdapter:
            from madrona_mjx_isaaclab.mesh_decimation.spec import spec_from_env, spec_from_level
            spec = spec_from_env(spec_from_level("medium"))
            return MeshGeometryAdapter(
                spec=spec,
                max_texture_size=resolve_max_texture_size(max_texture_size),
            )
        return adapter_class()

    @classmethod
    def register(cls, prim_type: str, adapter_class: type[GeometryAdapter]):
        """Register a new geometry adapter."""
        cls._adapters[prim_type] = adapter_class


def combine_geometries(geometries: list[SingleGeometry]) -> MadronaGeometry:
    """Combine multiple SingleGeometry into one MadronaGeometry.

    Note: Madrona expects vertex indices to be relative to each mesh (0-based per mesh),
    not global/absolute indices. So we do NOT add vertex_offset to indices.
    """
    if not geometries:
        return MadronaGeometry(
            vertices=np.zeros((0, 3), dtype=np.float32),
            normals=np.zeros((0, 3), dtype=np.float32),
            indices=np.zeros((0, 3), dtype=np.int32),
            uvs=np.zeros((0, 2), dtype=np.float32),
            colors=np.zeros((0, 4), dtype=np.float32),
            geom_vertex_offsets=[],
            geom_index_offsets=[],
            geom_uv_offsets=[],
            textures=[],
            geom_texture_ids=[],
        )

    all_vertices = []
    all_normals = []
    all_indices = []
    all_uvs = []
    all_colors = []
    vertex_offsets = []
    index_offsets = []
    uv_offsets = []
    textures = []
    geom_texture_ids = []

    # Texture deduplication: hash texture data to reuse identical textures
    texture_hash_to_id = {}  # hash -> texture index

    vertex_offset = 0
    index_offset = 0
    uv_offset = 0

    for geom in geometries:
        vertex_offsets.append(vertex_offset)
        index_offsets.append(index_offset)
        uv_offsets.append(uv_offset)

        all_vertices.append(geom.vertices)
        all_normals.append(geom.normals)
        # Keep indices relative to each mesh (0-based), as Madrona expects
        all_indices.append(geom.indices)
        all_colors.append(geom.color)

        # Handle UVs
        if geom.uvs is not None:
            all_uvs.append(geom.uvs)
            uv_offset += len(geom.uvs)
        else:
            # No UVs - use zeros
            dummy_uvs = np.zeros((len(geom.vertices), 2), dtype=np.float32)
            all_uvs.append(dummy_uvs)
            uv_offset += len(dummy_uvs)

        # Handle textures with deduplication
        if geom.texture is not None:
            # Create hash from texture dimensions and a sample of pixel data
            tex_hash = (
                geom.texture.width,
                geom.texture.height,
                (
                    geom.texture.pixels[0, 0].tobytes()
                    if geom.texture.pixels.size > 0
                    else b""
                ),
                (
                    geom.texture.pixels[-1, -1].tobytes()
                    if geom.texture.pixels.size > 0
                    else b""
                ),
            )

            if tex_hash in texture_hash_to_id:
                # Reuse existing texture
                tex_id = texture_hash_to_id[tex_hash]
                geom_texture_ids.append(tex_id)
            else:
                # New unique texture
                tex_id = len(textures)
                texture_hash_to_id[tex_hash] = tex_id
                geom_texture_ids.append(tex_id)
                textures.append(geom.texture)
        else:
            geom_texture_ids.append(-1)

        vertex_offset += len(geom.vertices)
        index_offset += len(geom.indices)

    return MadronaGeometry(
        vertices=np.concatenate(all_vertices),
        normals=np.concatenate(all_normals),
        indices=np.concatenate(all_indices),
        uvs=np.concatenate(all_uvs) if all_uvs else np.zeros((0, 2), dtype=np.float32),
        colors=np.array(all_colors),
        geom_vertex_offsets=vertex_offsets,
        geom_index_offsets=index_offsets,
        geom_uv_offsets=uv_offsets,
        textures=textures,
        geom_texture_ids=geom_texture_ids,
    )

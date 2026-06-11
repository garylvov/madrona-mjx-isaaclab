# madrona-mjx-isaaclab

Maintained standalone fork of [madrona_mjx](https://github.com/shacklettbp/madrona_mjx)
(the [Madrona](https://madrona-engine.github.io) GPU batch renderer + MuJoCo MJX
bindings), used as the high-throughput tiled-camera renderer for Isaac Lab.
Upstream is unmaintained; everything (engine + third-party deps) is vendored
here as plain files. Original commit provenance and the full list of changes:
[FORK.md](FORK.md). Upstream docs: [docs/README_upstream.md](docs/README_upstream.md).

## Install (prebuilt conda package)

```bash
pixi project channel add https://prefix.dev/garylvov
pixi add madrona-mjx
# or: conda install -c https://repo.prefix.dev/garylvov madrona-mjx
```

Requirements: linux-64, glibc >= 2.34 (Ubuntu 22.04+), NVIDIA driver for
CUDA 12.8, python 3.11. You must provide `mujoco >= 3.3.3` with mjx yourself
(intentionally not a dependency, to avoid clobbering pip-installed mujoco).

## Use

With Isaac Lab: point your scene at the Madrona camera configs
(`MadronaTiledCameraCfg` / `MadronaMultiCamTiledCameraCfg` in
[gigastrap](https://github.com/garylvov/gigastrap)'s `madrona_renderer`
extension). Texture resolution is set per camera config via
`max_texture_size` (falls back to the `MADRONA_MAX_TEXTURE_SIZE` env var,
then 512); all cameras of one renderer share geometry and texture memory.

With MJX directly:

```python
from madrona_mjx import BatchRenderer
renderer = BatchRenderer(mjx_model, gpu_id=0, num_worlds=N,
                         batch_render_view_width=64,
                         batch_render_view_height=64)
```

Smoke test: `python scripts/standalone_smoke.py`. Known issue: the pure-MJX
ECS engine init currently fails on sm_86 (RTX 30xx) with
`LAUNCH_OUT_OF_RESOURCES`; the Isaac Lab renderer path is unaffected.

## Build from source

```bash
pixi run build          # local dev build into build/
pixi run conda-build    # build the conda package (rattler-build)
pixi run conda-upload   # publish to prefix.dev/garylvov
```

No system toolchain needed; madrona fetches its own clang, the pixi env
supplies cmake/CUDA/sysroot. Notable fork changes beyond vendoring: sm_86
BVH `__launch_bounds__` fixes, CUDA-graph-safe `TmpAllocator` sizing, and
relocation-safe packaging (string-merge disabled, relocation-proof NVRTC
include paths) so the conda package works from any install prefix.

## License

MIT, same as upstream (see [LICENSE](LICENSE)).

# madrona-mjx-isaaclab

> **Disclaimer: not maintained.** This fork exists for our own projects and is
> published as-is. No support, no roadmap, no compatibility promises. Upstream
> madrona_mjx is also unmaintained, which is why this exists.

Standalone fork of [madrona_mjx](https://github.com/shacklettbp/madrona_mjx)
(the [Madrona](https://madrona-engine.github.io) GPU batch renderer + MuJoCo MJX
bindings), used as the high-throughput tiled-camera renderer for Isaac Lab.
Everything (engine + third-party deps) is vendored here as plain files.
Original commit provenance and the full list of changes: [FORK.md](FORK.md).
Upstream docs: [docs/README_upstream.md](docs/README_upstream.md).

## Install (prebuilt conda package)

```bash
pixi project channel add https://prefix.dev/garylvov
pixi add madrona-mjx-isaaclab
# or: conda install -c https://repo.prefix.dev/garylvov madrona-mjx-isaaclab
```

Requirements: linux-64, glibc >= 2.34 (Ubuntu 22.04+), NVIDIA driver for
CUDA 12.8, python 3.12 (canonical) or 3.11 (shipped for Isaac Sim envs,
which pin 3.11). Works with both old (0.5.x) and current jax. You must
provide `mujoco >= 3.3.3` with mjx yourself (intentionally not a dependency,
to avoid clobbering pip-installed mujoco).

If JAX runs in the same process (the MJX path does), set
`XLA_PYTHON_CLIENT_PREALLOCATE=false`: JAX's default ~75% GPU memory
preallocation starves Madrona's engine init on smaller GPUs (24GB RTX 30xx)
and it fails with `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`.

## Use with Isaac Lab

The conda package ships both the renderer (`madrona_mjx`) and the Isaac Lab
adapter (`madrona_mjx_isaaclab`: camera sensor + configs + USD scene-to-Madrona
geometry extraction). Drop a config into an `InteractiveSceneCfg` like any
other sensor. Madrona renders independently of Isaac's RTX camera pipeline,
so `--enable_cameras` is NOT required (headless works as usual):

```python
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils import configclass

from madrona_mjx_isaaclab.camera_cfg import (
    CameraOffsetCfg,
    MadronaCameraCfg,
    MadronaMultiCamTiledCameraCfg,
    MadronaTiledCameraCfg,
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # One camera per env; Madrona renders all envs in a single batched call.
    camera = MadronaTiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        width=128,
        height=128,
        data_types=["rgb", "depth"],
        spawn=PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955),
        offset=CameraCfg.OffsetCfg(
            pos=(1.0, 0.0, 0.5), rot=(0.61, 0.35, 0.35, 0.61), convention="world"
        ),
        use_rasterizer=True,   # False = raytracer
        max_texture_size=256,  # max edge of every ingested texture (0 disables downsampling)
    )
```

Frames arrive like any Isaac Lab camera: `env.scene["camera"].data.output["rgb"]`
with shape `(num_envs, H, W, 3)`, so standard observation terms work unchanged.

Multiple views per env render in the same batched call and share one
geometry/texture pool (`max_texture_size` applies to the whole sensor);
outputs are keyed `"rgb"`, `"rgb_1"`, ...:

```python
    cameras = MadronaMultiCamTiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/MultiCam",
        width=128,
        height=128,
        data_types=["rgb"],
        cameras=[
            # tracks an articulation link every frame
            MadronaCameraCfg(
                offset=CameraOffsetCfg(pos=(0.05, 0.0, 0.02), body_name="palm_link"),
                spawn=PinholeCameraCfg(focal_length=24.0),
            ),
            # static in the env-local frame
            MadronaCameraCfg(
                offset=CameraOffsetCfg(pos=(1.2, 0.0, 0.8)),
                spawn=PinholeCameraCfg(focal_length=24.0),
            ),
        ],
    )
```

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

## Local install into your own env (uv / pip)

The native build runs inside this repo's pixi env (your env stays clean);
the python side then installs editable into whatever env you use:

```bash
git clone https://github.com/garylvov/madrona-mjx-isaaclab && cd madrona-mjx-isaaclab
pixi run build        # compiles the renderer into build/
uv pip install -e .   # exposes madrona_mjx + madrona_mjx_isaaclab
```

Notes:
- Use a python 3.12 env (the extensions are built against the pixi env's 3.12).
- Editable-only: upstream's build backend emits a redirect into this checkout,
  so keep the clone around -- the compiled libs also resolve their CUDA
  libraries through it.
- Bring your own `jax` and `mujoco >= 3.3.3` (plus Isaac Lab if you use the
  `madrona_mjx_isaaclab` adapter).

## License

MIT, same as upstream (see [LICENSE](LICENSE)).

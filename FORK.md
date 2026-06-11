# Madrona fork (gigastrap)

Self-contained fork of `madrona_mjx` plus the vendored Madrona engine and all of its
third-party dependencies as plain files (no submodules). Upstream is unmaintained;
this tree is the source of truth going forward. It is built and published as the
`madrona-mjx` conda package on the `garylvov` prefix.dev channel.

## Provenance

Vendored from the gigastrap checkout on 2026-06-10. Base commits:

| Path (was submodule) | Upstream | Commit |
|---|---|---|
| `.` (madrona_mjx) | github.com/shacklettbp/madrona_mjx | `1b03bc893f34f1204c3190c883f01c8a989b3b94` |
| `external/madrona` | github.com/shacklettbp/madrona | `b46e6ab782cfd06956c35cb2ae42351a3fb5f38c` |
| `external/madrona/external/SPIRV-Reflect` | github.com/KhronosGroup/SPIRV-Reflect | `756e7b13243b5c4b110bb63dba72d10716dd1dfe` |
| `external/madrona/external/cccl` | github.com/NVIDIA/cccl | `0ebdddf0bf8fd2c69fbfbe4ee5049111faa6f226` |
| `external/madrona/external/fast_float` | github.com/fastfloat/fast_float | `69e0ea6f8ae521667ca80e41b4e8e44d8b33a31d` |
| `external/madrona/external/glfw` | github.com/glfw/glfw | `3fa2360720eeba1964df3c0ecf4b5df8648a8e52` |
| `external/madrona/external/googletest` | github.com/google/googletest | `dddb219c3eb96d7f9200f09b0a381f016e6b4562` |
| `external/madrona/external/imgui` | github.com/ocornut/imgui | `89d3dabf2e6a5a58c5d2ff3a82bbf4d2d478a07c` |
| `external/madrona/external/madrona-deps` | github.com/shacklettbp/madrona-deps | `8d57788474dffe3a011e2467df52f6285e662668` |
| `external/madrona/external/madrona-toolchain` | github.com/shacklettbp/madrona-toolchain | `8c0b55b52c74f2a2f237c97be332bd5d579a39c1` |
| `external/madrona/external/meshoptimizer` | github.com/zeux/meshoptimizer | `8764552531e55588a049b2d5f171db33200ac512` |
| `external/madrona/external/nanobind` | github.com/wjakob/nanobind | `0035bc43fa2af390b48608c9bf4595d7505c2386` |
| `external/madrona/external/nanobind/ext/robin_map` | github.com/Tessil/robin-map | `68ff7325b3898fca267a103bad5c509e8861144d` |
| `external/madrona/external/simdjson` | github.com/simdjson/simdjson | `412a8f7c4de85187e378f68b301c14c600c717b2` |
| `external/madrona/external/stb` | github.com/nothings/stb | `beebb24b945efdea3b9bba23affb8eb3ba8982e7` |
| `external/madrona/external/tinyusdz` | github.com/syoyo/tinyusdz | `b1d1b4719cd87636ff2fe463f47c2c61110b0966` |

Excluded from the vendor: `external/mujoco_menagerie` (958M, only used by demo
scripts; clone from google-deepmind/mujoco_menagerie @ `a88bc450470d97ebc1f282e5969675fb4d1f0ed7`
if a demo needs it).

## Gigastrap patches absorbed as commits

These used to live in `gigastrap/patches/` and were applied by the
`build-madrona-mjx` doit task at build time. They are now real commits here:

- `madrona_rt_launch_bounds.patch` — `__launch_bounds__` on the BVH kernels,
  `num_blocks_per_sm` 16 -> 4 (sm_86 scratchpad sizing), and `TmpAllocator` grow
  floor 1 MiB -> 256 MiB to keep `cuMemMap` out of captured CUDA graphs.
- `madrona_wayland_vsync.patch` — disable Wayland vsync in the viewer renderer.

`patches/madrona_cuda_device_fix.patch` was referenced by the build task but the
file never existed in the repo (the apply was `|| true`-silenced); there is nothing
to absorb.

## Build-time network fetches (longevity risk)

`external/madrona/external/madrona-deps` and `madrona-toolchain` FetchContent
prebuilt tarballs from shacklettbp's GitHub release pages at cmake configure time
(`bundled-deps/`, `bundled-toolchain/`, both gitignored here). If those releases
ever disappear, builds break. Consumers of the conda package are unaffected; only
rebuilds are. Mirroring the tarballs is an open follow-up.

## Building

This repo is fully standalone -- no Isaac, no gigastrap. The pixi env supplies
cmake/make, CUDA, a pinned sysroot (2.34, see below), and rattler-build:

```
pixi run build          # local dev build into build/
pixi run conda-build    # rattler-build the madrona-mjx conda package
pixi run conda-upload   # push to prefix.dev/garylvov
```

Notes:
- The sysroot works around hosts without `/usr/lib64/libm.so.6` (Ubuntu 24.04)
  and is pinned to 2.34: an unpinned sysroot resolves to 2.39 and leaks
  `__isoc23_*` (glibc 2.38) symbols into `libmadmjx_mgr.so`, making the conda
  package uninstallable on libc<2.39 hosts (clusters).
- The conda recipe builds from a source tree placed under `$PREFIX` so the
  absolute paths madrona bakes at configure time (runtime HLSL/sky data,
  NVRTC device sources, `DATA_DIR`) are prefix-relocated at install.
- `mujoco` is intentionally not a run dep of the conda package (gigastrap
  provides it via pip); consumers must supply `mujoco>=3.3.3` with mjx.

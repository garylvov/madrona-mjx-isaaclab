"""Standalone madrona-mjx smoke test: no Isaac, just mujoco-mjx + BatchRenderer.

Initializes the batch renderer on a trivial MJCF scene (one camera, one box)
and renders a frame. Exercises the full runtime path of the packaged build:
shader compilation (DXC), the NVRTC megakernel build, and the Vulkan/embree
library closure.

Run: python scripts/standalone_smoke.py [--rt]
"""

from __future__ import annotations

import argparse
import os
import sys

# JAX preallocates ~75% of GPU memory by default, which starves Madrona's
# launch-time memory reservation on smaller GPUs (e.g. 24GB RTX 30xx) and
# fails engine init with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import mujoco
import numpy as np
from mujoco import mjx

MJCF = """
<mujoco>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <camera name="cam" pos="0 -1 1" euler="45 0 0"/>
    <geom type="plane" size="2 2 0.1"/>
    <body pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rt", action="store_true", help="use the raytracer")
    parser.add_argument("--num-worlds", type=int, default=2)
    args = parser.parse_args()

    from madrona_mjx.renderer import BatchRenderer

    model = mujoco.MjModel.from_xml_string(MJCF)
    m = mjx.put_model(model)

    renderer = BatchRenderer(
        m,
        gpu_id=0,
        num_worlds=args.num_worlds,
        batch_render_view_width=64,
        batch_render_view_height=64,
        use_rasterizer=not args.rt,
    )

    d = mjx.make_data(model)
    d = jax.vmap(lambda _: d)(np.arange(args.num_worlds))
    state, rgb, depth = jax.vmap(renderer.init, in_axes=(0, None))(d, m)
    rgb = np.asarray(rgb)
    print(f"rendered rgb shape={rgb.shape} dtype={rgb.dtype} "
          f"min={rgb.min()} max={rgb.max()}")
    if rgb.shape[-3:-1] != (64, 64):
        print("FAIL: unexpected shape")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

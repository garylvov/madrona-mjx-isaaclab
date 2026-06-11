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

"""Persistent buffer pool for zero-copy GPU memory sharing between PyTorch and JAX.

This module provides a buffer pool that allocates GPU tensors once during initialization
and creates persistent JAX views via DLPack. This eliminates per-frame conversion overhead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.dlpack as jax_dlpack

# Import JAX modules at module level (not per-frame)
import torch
from torch.utils.dlpack import to_dlpack

if TYPE_CHECKING:
    from jax import Array


@dataclass(frozen=True)
class BufferSpec:
    """Immutable specification for a buffer."""

    shape: tuple[int, ...]
    dtype_torch: torch.dtype = torch.float32


class PersistentBufferPool:
    """Pre-allocated buffer pool for zero-copy PyTorch/JAX interop.

    Implements Flyweight pattern - allocates GPU buffers once during initialization
    and maintains persistent JAX views that can be reused every frame without
    DLPack conversion overhead.

    Usage:
        pool = PersistentBufferPool(device)
        pool.register("geom_pos", BufferSpec((num_envs, num_geoms, 3)))
        pool.register("geom_quat", BufferSpec((num_envs, num_geoms, 4)))
        pool.initialize()

        # Every frame (zero conversion cost):
        torch_tensor = pool.get_torch("geom_pos")
        torch_tensor.copy_(new_data)  # Update in-place
        jax_array = pool.get_jax("geom_pos")  # Same GPU memory, no conversion
    """

    def __init__(self, device: torch.device):
        """Initialize buffer pool.

        Args:
            device: PyTorch device for buffer allocation (should be CUDA).
        """
        self._device = device
        self._torch_buffers: dict[str, torch.Tensor] = {}
        self._jax_views: dict[str, Array] = {}
        self._specs: dict[str, BufferSpec] = {}
        self._initialized = False

    def register(self, name: str, spec: BufferSpec) -> None:
        """Register a buffer specification before initialization.

        Args:
            name: Unique identifier for this buffer.
            spec: Buffer specification (shape, dtype).

        Raises:
            RuntimeError: If called after initialize().
        """
        if self._initialized:
            raise RuntimeError("Cannot register buffers after initialization")
        self._specs[name] = spec

    def initialize(self) -> None:
        """Allocate all registered buffers and create persistent JAX views.

        This performs the one-time DLPack conversion cost. After this,
        get_jax() returns the pre-existing view with zero overhead.
        """
        if self._initialized:
            return

        # Ensure CUDA context is on correct device before allocations
        device = (
            torch.device(self._device)
            if isinstance(self._device, str)
            else self._device
        )
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device.index)

        for name, spec in self._specs.items():
            # Allocate contiguous PyTorch buffer
            self._torch_buffers[name] = torch.zeros(
                spec.shape,
                dtype=spec.dtype_torch,
                device=self._device,
            ).contiguous()

            # Create persistent JAX view via DLPack (one-time cost)
            self._jax_views[name] = jax_dlpack.from_dlpack(
                to_dlpack(self._torch_buffers[name])
            )

        self._initialized = True
        print(f"[PersistentBufferPool] Initialized {len(self._specs)} buffers")

    def get_torch(self, name: str) -> torch.Tensor:
        """Get PyTorch tensor for in-place updates.

        Args:
            name: Buffer identifier.

        Returns:
            PyTorch tensor that can be written to. Changes are visible
            to the corresponding JAX view immediately (same GPU memory).
        """
        if not self._initialized:
            raise RuntimeError("Buffer pool not initialized")
        return self._torch_buffers[name]

    def get_jax(self, name: str) -> Array:
        """Get JAX array view (zero conversion cost after init).

        Args:
            name: Buffer identifier.

        Returns:
            JAX array view of the same GPU memory as the PyTorch tensor.
        """
        if not self._initialized:
            raise RuntimeError("Buffer pool not initialized")
        return self._jax_views[name]

    @property
    def initialized(self) -> bool:
        """Whether the buffer pool has been initialized."""
        return self._initialized

    @property
    def buffer_names(self) -> list[str]:
        """List of registered buffer names."""
        return list(self._specs.keys())

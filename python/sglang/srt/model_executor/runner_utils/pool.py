# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Process-wide CUDA graph memory pool shared across the prefill and
decode graph backends. The two phases never replay concurrently, so
sharing one pool reserves only the larger phase's capture footprint.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.runtime_context import get_resources


def get_global_graph_memory_pool() -> Optional[Any]:
    return get_resources().graph_memory_pool


def set_global_graph_memory_pool(val: Any) -> None:
    resources = get_resources()
    owner = resources.graph_memory_pool_owner
    if owner is not None and owner.id != val:
        resources.graph_memory_pool_owner = None
        resources.graph_memory_pool_owner_device = None
    resources.graph_memory_pool = val


def get_global_graph_memory_pool_owner(pool: Any, device: Any) -> Optional[Any]:
    """Only expose an owner created here for this handle and CUDA device.

    An externally installed legacy handle cannot be adopted as a MemPool via
    the public API. Such handles retain their original capture-only behavior.
    """
    resources = get_resources()
    owner = resources.graph_memory_pool_owner
    if (
        owner is not None
        and owner.id == pool == resources.graph_memory_pool
        and getattr(device, "type", None) == "cuda"
        and device.index == resources.graph_memory_pool_owner_device
    ):
        return owner
    return None


def get_or_create_global_graph_memory_pool(
    device_module: Any, *, allow_explicit_pool: bool = True
) -> Any:
    """Return the shared graph memory pool, creating it on first use so
    later backends reuse the same handle."""
    resources = get_resources()
    if resources.graph_memory_pool is None:
        import torch

        if (
            allow_explicit_pool
            and device_module is torch.cuda
            and hasattr(torch.cuda, "MemPool")
            and hasattr(torch.cuda, "use_mem_pool")
            and torch.cuda.memory.get_allocator_backend() == "native"
        ):
            owner = torch.cuda.MemPool()
            resources.graph_memory_pool_owner = owner
            resources.graph_memory_pool_owner_device = torch.cuda.current_device()
            resources.graph_memory_pool = owner.id
        else:
            resources.graph_memory_pool = device_module.graph_pool_handle()
    return resources.graph_memory_pool


def release_unused_global_graph_memory_pool(pool: Any) -> None:
    """Release our cache only after the last graph/context releases its pool.

    Full graphs also keep a Python owner lease, so replacing the global handle
    cannot invalidate a graph retained by another caller.
    """
    resources = get_resources()
    owner = resources.graph_memory_pool_owner
    if owner is None or owner.id != pool or resources.graph_memory_pool != pool:
        return
    import torch

    with torch.cuda.device(resources.graph_memory_pool_owner_device):
        if owner.use_count() == 1:
            set_global_graph_memory_pool(None)

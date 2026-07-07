from __future__ import annotations

from typing import Callable, List, Optional

import torch


def compute_group_size(num_layers: int, avg_tokens: int) -> int:
    """Adaptive group_size based on prompt length and model depth.

    Short prompts need more groups (higher T/C ratio).
    Long prompts need fewer groups (compute dominates).
    Returns the base group_size for progressive scheduling.
    """
    from sglang.srt.environ import envs

    fixed = envs.SGLANG_PIPELINE_GROUP_SIZE.get()
    if fixed > 0:
        return min(fixed, num_layers)

    min_tokens = envs.SGLANG_PIPELINE_MIN_TOKENS.get()
    max_iters = envs.SGLANG_PIPELINE_MAX_ITERS.get()
    min_iters = envs.SGLANG_PIPELINE_MIN_ITERS.get()

    sat_tokens = min_tokens * 3
    t = max(0.0, min(1.0, (avg_tokens - min_tokens) / max(1, sat_tokens - min_tokens)))
    target_iters = max(min_iters, min(max_iters, max_iters - t * (max_iters - min_iters)))
    group_size = max(1, num_layers // int(target_iters))
    return group_size


def build_progressive_groups(num_layers: int, base_group_size: int) -> List[int]:
    """Build progressive group boundaries: first group small, then ramp up.

    Strategy: first group = 1 layer (trigger decode ASAP), second group = 2 layers,
    then ramp up to base_group_size for remaining groups.
    """
    if base_group_size <= 1:
        return list(range(1, num_layers + 1))

    boundaries = []
    pos = 0
    # First group: 1 layer (fastest TTFT)
    pos += 1
    boundaries.append(pos)
    if pos >= num_layers:
        return boundaries
    # Second group: min(2, base_group_size // 2)
    second = max(1, min(2, base_group_size // 2))
    pos += second
    pos = min(pos, num_layers)
    boundaries.append(pos)
    if pos >= num_layers:
        return boundaries
    # Remaining: base_group_size each
    while pos < num_layers:
        pos = min(pos + base_group_size, num_layers)
        boundaries.append(pos)
    return boundaries


class LayerKVReadyCollector:
    def __init__(self, num_layers: int):
        # blocking=True -> cudaEventBlockingSync: the transfer worker's
        # cuda_event.synchronize() (mooncake/conn.py) sleep-waits instead of
        # busy-spinning a CPU core. Spin-sync stole a core from the prefill
        # main thread's kernel-launch loop, leaking into forward (+~89ms in
        # R-reuse-3 stage decomposition). Sleep-wait frees the core so the
        # per-layer send truly overlaps forward compute instead of stalling it.
        self.events: List[torch.cuda.Event] = [
            torch.cuda.Event(blocking=True) for _ in range(num_layers)
        ]
        self.ready = [False] * num_layers

    def record_layer_ready(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= len(self.events):
            return
        self.events[layer_id].record()
        self.ready[layer_id] = True

    def get_event(self, layer_id: int) -> Optional[torch.cuda.Event]:
        if layer_id < 0 or layer_id >= len(self.events) or not self.ready[layer_id]:
            return None
        return self.events[layer_id]

    def all_ready(self) -> bool:
        return all(self.ready)


class LayerTransferCounter:
    """Per-request RDMA receive completion tracker for decode per-layer compute."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(num_layers)
        ]
        self.completed = [False] * num_layers

    @property
    def first_layer_ready(self) -> bool:
        return self.completed[0]

    def complete(self, layer_id: int):
        """Called by RDMA receiver when layer_id's data lands on GPU."""
        if 0 <= layer_id < self.num_layers:
            self.events[layer_id].record()
            self.completed[layer_id] = True

    def wait_until(self, layer_id: int):
        """Block current CUDA stream until layer_id transfer is done."""
        if 0 <= layer_id < self.num_layers:
            torch.cuda.current_stream().wait_event(self.events[layer_id])


class LayerKVDispatcher:
    """Inline dispatcher: dispatch send_layer from main thread at group boundaries."""

    def __init__(
        self,
        collector: LayerKVReadyCollector,
        reqs,
        page_indices_per_req: List,
        group_boundaries: List[int],
        num_layers: int,
        send_layer_fn: Callable,
        send_final_metadata_fn: Callable,
    ):
        self.collector = collector
        self.reqs = reqs
        self.page_indices_per_req = page_indices_per_req
        self.group_boundaries = group_boundaries
        self.num_layers = num_layers
        self.send_layer_fn = send_layer_fn
        self.send_final_metadata_fn = send_final_metadata_fn
        self.next_group_idx = 0
        self.next_layer = 0
        self.all_dispatched = False

    def try_dispatch(self, layer_id: int):
        """Called from main thread after record_layer_ready. Non-blocking RDMA enqueue."""
        if self.all_dispatched or self.next_group_idx >= len(self.group_boundaries):
            return
        boundary = self.group_boundaries[self.next_group_idx]
        if layer_id + 1 < boundary:
            return
        for lid in range(self.next_layer, boundary):
            for req_idx, req in enumerate(self.reqs):
                self.send_layer_fn(
                    req,
                    self.page_indices_per_req[req_idx],
                    lid,
                    self.collector.events[lid],
                    lid == self.num_layers - 1,
                )
        self.next_layer = boundary
        self.next_group_idx += 1
        if self.next_layer >= self.num_layers:
            # Final metadata carries the sampled output token (req.output_ids[0]),
            # which does not exist until after the forward pass + sampling. Defer
            # send_final_metadata to process_batch_result_disagg_prefill (after
            # req.output_ids.append). Here we only mark all per-layer KV dispatched.
            self.all_dispatched = True

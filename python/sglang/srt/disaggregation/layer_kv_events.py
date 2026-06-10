from __future__ import annotations

from typing import List, Optional

import torch


class LayerKVReadyCollector:
    def __init__(self, num_layers: int):
        self.events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(num_layers)
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

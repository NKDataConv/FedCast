from __future__ import annotations

import torch
import torch.nn as nn

from fedcast.datasets.dataset_sinus import WINDOW_SIZE


class LinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(WINDOW_SIZE, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(x)



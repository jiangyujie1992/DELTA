"""Query-guided temporal residual for projected video and query sequences."""

import torch
from torch import nn
from torch.nn import functional as F


class DeltaResidual(nn.Module):
    def __init__(self, dim, variant="delta"):
        super().__init__()
        if variant not in {"delta", "static", "qoff", "shifted"}:
            raise ValueError(f"Unknown residual variant: {variant}")
        self.variant = variant
        self.proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, video, query, video_mask, query_mask):
        """Both masks use ones for valid positions and zeros for padding."""
        vm = video_mask.to(video.dtype)
        qm = query_mask.to(query.dtype)
        pooled = (query * qm.unsqueeze(-1)).sum(1) / qm.sum(1, keepdim=True).clamp_min(
            1
        )
        if self.variant == "static":
            signal = video * vm.unsqueeze(-1)
        else:
            previous = F.pad(video[:, :-1], (0, 0, 1, 0))
            valid_pair = vm * F.pad(vm[:, :-1], (1, 0))
            signal = (video - previous) * valid_pair.unsqueeze(-1)
            if self.variant == "shifted":
                signal = shift_valid_differences(signal, vm)
        if self.variant == "qoff":
            pooled = torch.zeros_like(pooled)
        update = F.gelu(self.proj(signal)) * torch.sigmoid(self.gate(pooled)).unsqueeze(
            1
        )
        return video + torch.tanh(self.scale) * update * vm.unsqueeze(-1)


def shift_valid_differences(signal, mask):
    """Rotate only the differences between consecutive valid video clips."""
    lengths = mask.bool().sum(1)
    index = torch.arange(signal.shape[1], device=signal.device)[None, :]
    if not torch.equal(mask.bool(), index < lengths[:, None]):
        raise ValueError("The shifted control requires contiguous prefix masks.")
    count = (lengths - 1).clamp_min(1)[:, None]
    offset = (lengths - 1).clamp_min(0).div(2, rounding_mode="floor")[:, None]
    rotated = (index - 1 - offset).remainder(count) + 1
    source = torch.where((index >= 1) & (index < lengths[:, None]), rotated, index)
    return signal.gather(1, source.unsqueeze(-1).expand_as(signal))

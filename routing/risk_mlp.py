"""Matched, frozen Task 4 risk predictors; no FastWAM import or policy loading.

Suite IDs are fixed: libero_spatial=0 and libero_goal=1. Input coordinates are
1024 pooled L4 hidden values, ten step indicators, then two suite indicators.
The mask is applied *after* standardization, so excluded coordinates are zero.
"""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


SUITES = ("libero_spatial", "libero_goal")
MODES = ("content", "timestep", "meta")
ANCHORS = (0, 4, 9)
ELIGIBLE = (1, 2, 3, 5, 6, 7, 8)
INPUT_DIM = 1036
ADAPTER_SHA = "3bfe747be64de3cc92b5ba3868d3a3f9450aa1532a77ce6574f3023b8eb84c58"


def make_features(pooled_hidden, steps, suite_ids):
    """Return float32 [..., 1036] features on pooled_hidden.device.

    ``pooled_hidden`` is already pooled, with last dimension 1024. ``steps``
    and ``suite_ids`` are tensors broadcastable to its leading dimensions.
    This function contains no host synchronization and can be torch.compiled.
    """
    shape = pooled_hidden.shape[:-1]
    step_ids = steps.to(device=pooled_hidden.device, dtype=torch.long).expand(shape)
    suite_ids = suite_ids.to(device=pooled_hidden.device, dtype=torch.long).expand(shape)
    return torch.cat((pooled_hidden.float(), F.one_hot(step_ids, 10).float(),
                      F.one_hot(suite_ids, len(SUITES)).float()), dim=-1)


class RiskMLP(nn.Module):
    """Identical 1036 -> 64 SiLU -> 1 architecture for each input ablation."""

    def __init__(self, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown predictor {mode!r}; expected {MODES}")
        self.mode = mode
        self.register_buffer("mean", torch.zeros(INPUT_DIM))
        self.register_buffer("std", torch.ones(INPUT_DIM))
        mask = torch.zeros(INPUT_DIM)
        mask[1024:1034] = 1
        if mode == "content":
            mask[:1024] = 1
        elif mode == "meta":
            mask[1034:] = 1
        self.register_buffer("mask", mask)
        self.net = nn.Sequential(nn.Linear(INPUT_DIM, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, values):
        values = ((values.float() - self.mean) / self.std) * self.mask
        return self.net(values).squeeze(-1)


def load_router(run, mode, device="cpu"):
    """Load run/routers/<mode>/best.pt and return an eval, frozen RiskMLP.

    The caller owns inference compilation. No policy or other model is loaded.
    """
    path = Path(run) / "routers" / mode / "best.pt"
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("mode") != mode or payload.get("input_dim") != INPUT_DIM:
        raise ValueError(f"Predictor mode or input dimension mismatch: {path}")
    if payload.get("adapter_sha256") != ADAPTER_SHA:
        raise ValueError(f"Frozen fork4_b1 identity mismatch: {path}")
    router = RiskMLP(mode).to(device)
    expected_mask = router.mask.detach().clone()
    router.load_state_dict(payload["state"], strict=True)
    if not torch.equal(router.mask, expected_mask):
        raise ValueError(f"Checkpoint contains the wrong input mask: {path}")
    if not bool(torch.isfinite(router.mean).all()) or not bool(torch.isfinite(router.std).all()):
        raise ValueError(f"Invalid normalization buffers: {path}")
    if not bool((router.std > 0).all()):
        raise ValueError(f"Nonpositive normalization scale: {path}")
    return router.eval().requires_grad_(False)

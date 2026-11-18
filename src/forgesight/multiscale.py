"""Validated fusion of two ResNet spatial feature levels."""

import torch
import torch.nn.functional as F


def fuse_resnet_stages(
    stage2: torch.Tensor,
    stage3: torch.Tensor,
) -> torch.Tensor:
    """Combine fine and coarse CNN features on the finer spatial grid."""

    if not isinstance(stage2, torch.Tensor):
        raise ValueError("Stage 2 must be a tensor.")

    if not isinstance(stage3, torch.Tensor):
        raise ValueError("Stage 3 must be a tensor.")

    if (
        stage2.ndim != 4
        or stage3.ndim != 4
        or stage2.shape[0] != stage3.shape[0]
        or stage2.shape[1] != 128
        or stage3.shape[1] != 256
        or min(stage2.shape[-2:]) <= 0
        or min(stage3.shape[-2:]) <= 0
    ):
        raise ValueError("Unexpected ResNet feature-map dimensions.")

    if (
        not torch.isfinite(stage2).all().item()
        or not torch.isfinite(stage3).all().item()
    ):
        raise ValueError("Feature maps contain nonfinite values.")

    # Local averaging supplies neighborhood context.
    fine = F.avg_pool2d(stage2, kernel_size=3, stride=1, padding=1)
    coarse = F.avg_pool2d(stage3, kernel_size=3, stride=1, padding=1)

    # Bring the coarse features onto the finer spatial grid.
    coarse = F.interpolate(
        coarse,
        size=fine.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    # Concatenate feature channels, not spatial dimensions.
    fused = torch.cat((fine, coarse), dim=1)

    if fused.shape[1] != 384:
        raise ValueError("Unexpected fused feature dimension.")

    return fused


def spatial_feature_array(fused: torch.Tensor):
    """Convert one CNN feature tensor into H x W x C NumPy features."""

    if (
        not isinstance(fused, torch.Tensor)
        or fused.ndim != 4
        or fused.shape[0] != 1
        or fused.shape[1] != 384
    ):
        raise ValueError("Expected one fused 384-channel feature map.")

    return (
        fused[0]
        .detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .contiguous()
        .numpy()
    )

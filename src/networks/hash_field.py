# Instant-NGP 3D multi-resolution hash encoding.

import math
import torch
import torch.nn as nn
import tinycudann as tcnn


class HashField(nn.Module):
    """
    Instant-NGP-style 3D hash encoding over a world-space axis-aligned bound.
    Normalizes world coords to [0, 1] internally, then calls tcnn.Encoding.
    Output feature dim is n_levels * n_features_per_level.
    """
    def __init__(self, bound, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16, finest_resolution=2048):
        super().__init__()

        self.register_buffer('bound', bound.clone().float())
        self.out_dim = n_levels * n_features_per_level

        per_level_scale = math.exp(
            (math.log(finest_resolution) - math.log(base_resolution)) / max(n_levels - 1, 1)
        )

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
                "interpolation": "Linear",
            },
            dtype=torch.float32,
        )

    def forward(self, p):
        in_shape = p.shape
        p = p.reshape(-1, 3)
        lo = self.bound[:, 0]
        hi = self.bound[:, 1]
        p_nor = (p - lo) / (hi - lo)
        p_nor = p_nor.clamp(0.0, 1.0).contiguous()
        feat = self.encoding(p_nor)
        return feat.reshape(*in_shape[:-1], self.out_dim)

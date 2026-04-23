# This file is a part of SNI-SLAM.
# Originally derived from ESLAM (Apache-2.0). Tri-plane features have been
# replaced with Instant-NGP 3D hash encodings.
# MLP layers use tinycudann FullyFusedMLP for reduced latency.

import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn

from src.networks.hash_field import HashField


def _fused_mlp(n_input_dims, n_neurons, n_hidden_layers, output_activation="None"):
    """Build a FullyFusedMLP. n_neurons must be in {16, 32, 64, 128}."""
    return tcnn.Network(
        n_input_dims=n_input_dims,
        n_output_dims=n_neurons,
        network_config={
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": output_activation,
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
        },
    )


class Decoders(nn.Module):
    def __init__(self, c_dim=16, hidden_dim=32, truncation=0.08, n_blocks=2, learnable_beta=True,
                 num_classes=25, sem_hidden_dim=128, fused_hidden_dim=16,
                 bound=None, hash_cfg=None):
        super().__init__()

        self.c_dim = c_dim
        self.truncation = truncation
        self.n_blocks = n_blocks

        assert bound is not None, "Decoders requires the scene bound to size the hash grids."
        assert hash_cfg is not None, "Decoders requires a hash_cfg dict."
        feat_dim = 2 * c_dim  # preserves downstream MLP shapes
        self._build_hash_fields(bound, feat_dim, hash_cfg)

        # SDF trunk: feat_dim → hidden_dim (ReLU out), then linear head → 1
        self.linears = _fused_mlp(feat_dim, hidden_dim, n_blocks, output_activation="ReLU")
        self.output_linear = nn.Linear(hidden_dim, 1)

        # Semantic trunk: feat_dim → sem_hidden_dim (must be ≤128 for FullyFusedMLP)
        assert sem_hidden_dim in (16, 32, 64, 128), \
            f"sem_hidden_dim must be 16/32/64/128 for FullyFusedMLP, got {sem_hidden_dim}"
        self.s_linears = _fused_mlp(feat_dim, sem_hidden_dim, 3, output_activation="ReLU")
        self.s_output_linear = nn.Linear(sem_hidden_dim, num_classes)

        # Fused SDF+RGB trunk: feat_dim → fused_hidden_dim, then split head
        self.fused_linear = _fused_mlp(feat_dim, fused_hidden_dim, 4, output_activation="ReLU")
        self.out_sdf = nn.Linear(fused_hidden_dim, fused_hidden_dim + 1)

        # RGB head: (fused_hidden_dim + 2*feat_dim) → fused_hidden_dim → 3
        rgb_in_dim = fused_hidden_dim + 2 * feat_dim
        self.out_rgb = nn.Sequential(
            nn.Linear(rgb_in_dim, fused_hidden_dim),
            nn.ReLU(),
            nn.Linear(fused_hidden_dim, 3))

        if learnable_beta:
            self.beta = nn.Parameter(10 * torch.ones(1))
            self.semantic_beta = nn.Parameter(10 * torch.ones(1))
        else:
            self.beta = 10

    def _build_hash_fields(self, bound, feat_dim, hash_cfg):
        n_features_per_level = hash_cfg['n_features_per_level']
        n_levels = feat_dim // n_features_per_level
        assert n_levels * n_features_per_level == feat_dim, \
            f"2*c_dim ({feat_dim}) must be divisible by n_features_per_level ({n_features_per_level})"

        def make(role):
            role_cfg = hash_cfg.get(role, {})
            return HashField(
                bound=bound,
                n_levels=n_levels,
                n_features_per_level=n_features_per_level,
                log2_hashmap_size=role_cfg.get('log2_hashmap_size', hash_cfg['log2_hashmap_size']),
                base_resolution=role_cfg.get('base_resolution', hash_cfg['base_resolution']),
                finest_resolution=role_cfg.get('finest_resolution', hash_cfg['finest_resolution']),
            )

        self.hash_sdf = make('sdf')
        self.hash_color = make('color')
        self.hash_semantic = make('semantic')

    @property
    def bound(self):
        return self.hash_sdf.bound

    @bound.setter
    def bound(self, value):
        value = value.to(self.hash_sdf.bound.device).float()
        self.hash_sdf.bound.copy_(value)
        self.hash_color.bound.copy_(value)
        self.hash_semantic.bound.copy_(value)

    def get_raw_sdf(self, p):
        feat = self.hash_sdf(p).reshape(-1, 2 * self.c_dim)
        h = self.linears(feat).float()
        sdf = torch.tanh(self.output_linear(h)).squeeze(-1)
        return sdf, feat

    def get_raw_semantic(self, p):
        s_feat = self.hash_semantic(p).reshape(-1, 2 * self.c_dim)
        h = self.s_linears(s_feat).float()
        semantic = self.s_output_linear(h)
        return semantic, s_feat

    def get_raw_sdf_rgb(self, p):
        feat = self.hash_sdf(p).reshape(-1, 2 * self.c_dim)
        c_feat = self.hash_color(p).reshape(-1, 2 * self.c_dim)
        s_feat = self.hash_semantic(p).reshape(-1, 2 * self.c_dim)

        h = self.fused_linear(feat).float()
        sdf_out = self.out_sdf(h)
        sdf = torch.tanh(sdf_out[:, :1]).squeeze(-1)
        sdf_feat = sdf_out[:, 1:]

        rgb_in = torch.cat([sdf_feat, c_feat, s_feat], dim=-1)
        rgb = torch.sigmoid(self.out_rgb(rgb_in))

        return sdf, rgb, feat, c_feat

    def forward(self, p):
        p_shape = p.shape
        sdf, rgb, sdf_feat, rgb_feat = self.get_raw_sdf_rgb(p)
        semantic, semantic_feat = self.get_raw_semantic(p)
        raw = torch.cat([rgb, sdf.unsqueeze(-1), semantic], dim=-1)
        plane_feat = torch.cat(
            [rgb_feat[:, self.c_dim:], sdf_feat[:, self.c_dim:], semantic_feat[:, self.c_dim:]], dim=-1)
        raw = raw.reshape(*p_shape[:-1], -1)
        return raw, plane_feat

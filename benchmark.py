"""
SNI-SLAM Performance Benchmark
===============================
Measures speed of each pipeline stage in isolation so you can compare
before/after optimizations without running the full 2000-frame sequence.

Uses the first 20 frames of a Replica scene.  Total runtime: ~2-5 min on GPU.

Usage:
    pixi run python benchmark.py configs/Replica/room1.yaml

Output:
    Prints a table of per-stage timings + throughput numbers.
    Saves results to <output>/benchmark_results.json for diff across runs.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src import config
from src.common import get_samples, get_rays, matrix_to_cam_pose, cam_pose_to_matrix
from src.utils.datasets import get_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cuda_timer(fn, warmup=3, repeats=10):
    """Time a CUDA function with proper synchronization. Returns ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


# ---------------------------------------------------------------------------
# Benchmark stages
# ---------------------------------------------------------------------------

def bench_hash_field_forward(decoders, bound, device):
    """Raw hash grid lookup throughput."""
    n_points = 500_000
    pts = torch.rand(n_points, 3, device=device)
    lo = bound[:, 0].to(device)
    hi = bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)

    def fn():
        with torch.no_grad():
            _ = decoders.hash_sdf(pts)

    stats = cuda_timer(fn)
    stats["n_points"] = n_points
    stats["throughput_Mpts_s"] = round(n_points / stats["mean_ms"] / 1000, 2)
    return stats


def bench_decoder_forward(decoders, bound, device):
    """Full decoder forward (all 3 hash grids + MLPs)."""
    n_points = 200_000
    pts = torch.rand(n_points, 3, device=device)
    lo = bound[:, 0].to(device)
    hi = bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)

    def fn():
        with torch.no_grad():
            _ = decoders(pts)

    stats = cuda_timer(fn)
    stats["n_points"] = n_points
    stats["throughput_Mpts_s"] = round(n_points / stats["mean_ms"] / 1000, 2)
    return stats


def bench_decoder_backward(decoders, bound, device):
    """Decoder forward + backward (simulates one mapping optimization step)."""
    n_points = 50_000
    pts = torch.rand(n_points, 3, device=device)
    lo = bound[:, 0].to(device)
    hi = bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)

    optimizer = torch.optim.Adam(decoders.parameters(), lr=1e-3)

    def fn():
        optimizer.zero_grad()
        raw, _ = decoders(pts)
        loss = raw.sum()
        loss.backward()
        optimizer.step()

    stats = cuda_timer(fn, warmup=3, repeats=10)
    stats["n_points"] = n_points
    return stats


def bench_render_batch(renderer, decoders, frame_data, cfg, device):
    """Render a batch of rays (the core inner loop of tracking + mapping)."""
    _, gt_color, gt_depth, gt_c2w, _ = frame_data
    gt_color = gt_color.to(device)
    gt_depth = gt_depth.to(device)
    gt_c2w = gt_c2w.to(device)

    H, W = cfg['cam']['H'], cfg['cam']['W']
    fx, fy = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy = cfg['cam']['cx'], cfg['cam']['cy']
    truncation = cfg['model']['truncation']
    c_dim = cfg['model']['c_dim']
    n_rays = cfg['mapping']['pixels']

    c2w = gt_c2w.unsqueeze(0)

    def fn():
        with torch.no_grad():
            rays_o, rays_d, batch_depth, batch_color, _, _, _ = get_samples(
                0, H, 0, W, n_rays, H, W, fx, fy, cx, cy,
                c2w, gt_depth.unsqueeze(0), gt_color.unsqueeze(0),
                device=device, dim=c_dim)
            renderer.render_batch_ray(
                decoders, rays_d, rays_o, device, truncation, gt_depth=batch_depth)

    stats = cuda_timer(fn, warmup=2, repeats=8)
    stats["n_rays"] = n_rays
    stats["throughput_Krays_s"] = round(n_rays / stats["mean_ms"], 2)
    return stats


def bench_render_full_image(renderer, decoders, frame_data, cfg, device):
    """Render a complete H×W image (used for visualization / meshing)."""
    _, gt_color, gt_depth, gt_c2w, _ = frame_data
    gt_depth = gt_depth.to(device)
    gt_c2w = gt_c2w.to(device)
    truncation = cfg['model']['truncation']

    def fn():
        with torch.no_grad():
            renderer.render_img(decoders, gt_c2w, truncation, device, gt_depth=gt_depth)

    stats = cuda_timer(fn, warmup=1, repeats=3)
    H, W = cfg['cam']['H'], cfg['cam']['W']
    stats["resolution"] = f"{H}x{W}"
    stats["total_rays"] = H * W
    return stats


def bench_mapping_iteration(decoders, renderer, model_manager, frame_data, cfg, device):
    """One full mapping optimization iteration (sample rays + render + loss + backward)."""
    _, gt_color, gt_depth, gt_c2w, gt_semantic = frame_data
    gt_color = gt_color.to(device)
    gt_depth = gt_depth.to(device)
    gt_c2w = gt_c2w.to(device)
    gt_semantic = gt_semantic.to(device)

    H, W = cfg['cam']['H'], cfg['cam']['W']
    fx, fy = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy = cfg['cam']['cx'], cfg['cam']['cy']
    truncation = cfg['model']['truncation']
    c_dim = cfg['model']['c_dim']
    n_rays = cfg['mapping']['pixels']

    c2w = gt_c2w.unsqueeze(0)

    # Pre-compute CNN features (as Mapper.run does)
    with torch.no_grad():
        frame_rgb = gt_color.permute(2, 0, 1).unsqueeze(0).to(device)
        model_manager.set_mode_feature()
        sem_feat = model_manager.cnn(frame_rgb)
        rgb_feat = model_manager.head(sem_feat)
        sem_feat_sq = sem_feat.squeeze(0)
        rgb_feat_sq = rgb_feat.squeeze(0)

    kf_sem = sem_feat_sq.unsqueeze(0)
    kf_rgb = rgb_feat_sq.unsqueeze(0)

    model_manager.set_mode_result()
    with torch.no_grad():
        gt_sem_label = model_manager.cnn(frame_rgb)
    kf_gt_label = gt_sem_label.unsqueeze(0)

    optimizer = torch.optim.Adam(decoders.parameters(), lr=1e-3)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    w_color = cfg['mapping']['w_color']
    w_depth = cfg['mapping']['w_depth']
    w_semantic = cfg['mapping']['w_semantic']
    w_feature = cfg['mapping']['w_feature']
    w_sdf_fs = cfg['mapping']['w_sdf_fs']
    w_sdf_center = cfg['mapping']['w_sdf_center']
    w_sdf_tail = cfg['mapping']['w_sdf_tail']

    def fn():
        optimizer.zero_grad()

        rays_o, rays_d, batch_depth, batch_color, batch_sem, batch_rgb, batch_label = get_samples(
            0, H, 0, W, n_rays, H, W, fx, fy, cx, cy,
            c2w, gt_depth.unsqueeze(0), gt_color.unsqueeze(0),
            sem_feats=kf_sem, rgb_feats=kf_rgb, gt_label=kf_gt_label,
            device=device, dim=c_dim)

        depth, color, sdf, z_vals, gt_feat, plane_feat, render_semantic = renderer.render_batch_ray(
            decoders, rays_d, rays_o, device, truncation,
            gt_depth=batch_depth, sem_feats=batch_sem, rgb_feats=batch_rgb,
            return_emb=True)

        depth_mask = (batch_depth > 0)

        # SDF loss (simplified — uses free-space only for speed)
        front_mask = (z_vals < (batch_depth[:, None] - truncation)).bool()
        fs_loss = w_sdf_fs * torch.mean(torch.square(sdf[front_mask] - 1.0))

        color_loss = w_color * torch.square(batch_color - color).mean()
        depth_loss = w_depth * torch.square(batch_depth[depth_mask] - depth[depth_mask]).mean()

        feature_loss = w_feature * (gt_feat.detach() - plane_feat.detach()).abs().mean()
        semantic_loss = w_semantic * ce_loss_fn(render_semantic, batch_label)

        loss = fs_loss + color_loss + depth_loss + feature_loss + semantic_loss
        loss.backward()
        optimizer.step()

    stats = cuda_timer(fn, warmup=2, repeats=8)
    stats["n_rays"] = n_rays
    return stats


def bench_tracking_iteration(decoders, renderer, frame_data, cfg, device):
    """One tracking optimization iteration (pose refinement)."""
    _, gt_color, gt_depth, gt_c2w, gt_semantic = frame_data
    gt_color = gt_color.to(device)
    gt_depth = gt_depth.to(device)
    gt_c2w = gt_c2w.to(device)

    H, W = cfg['cam']['H'], cfg['cam']['W']
    fx, fy = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy = cfg['cam']['cx'], cfg['cam']['cy']
    truncation = cfg['model']['truncation']
    c_dim = cfg['model']['c_dim']
    n_rays = cfg['tracking']['pixels']
    ignore_edge_H = cfg['tracking']['ignore_edge_H']
    ignore_edge_W = cfg['tracking']['ignore_edge_W']

    cam_pose = matrix_to_cam_pose(gt_c2w.unsqueeze(0))
    T = torch.nn.Parameter(cam_pose[:, -3:].clone())
    R = torch.nn.Parameter(cam_pose[:, :4].clone())
    optimizer = torch.optim.Adam([
        {'params': [T], 'lr': cfg['tracking']['lr_T'], 'betas': (0.5, 0.999)},
        {'params': [R], 'lr': cfg['tracking']['lr_R'], 'betas': (0.5, 0.999)},
    ])

    decoders_frozen = copy.deepcopy(decoders)
    for p in decoders_frozen.parameters():
        p.requires_grad_(False)

    def fn():
        cam = torch.cat([R, T], -1)
        c2w = cam_pose_to_matrix(cam)

        rays_o, rays_d, batch_depth, batch_color, _, _, _ = get_samples(
            ignore_edge_H, H - ignore_edge_H,
            ignore_edge_W, W - ignore_edge_W,
            n_rays, H, W, fx, fy, cx, cy,
            c2w, gt_depth.unsqueeze(0), gt_color.unsqueeze(0),
            device=device, dim=c_dim)

        depth, color, sdf, z_vals, _, _, _ = renderer.render_batch_ray(
            decoders_frozen, rays_d, rays_o, device, truncation, gt_depth=batch_depth)

        depth_mask = (batch_depth > 0)
        loss = torch.square(batch_depth[depth_mask] - depth[depth_mask]).mean()
        loss = loss + 5.0 * torch.square(batch_color - color).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    stats = cuda_timer(fn, warmup=2, repeats=8)
    stats["n_rays"] = n_rays
    return stats


def bench_cnn_feature_extraction(model_manager, frame_data, cfg, device):
    """DINOv2 CNN feature extraction for one frame."""
    _, gt_color, _, _, _ = frame_data
    gt_color = gt_color.to(device)
    frame_rgb = gt_color.permute(2, 0, 1).unsqueeze(0)

    def fn():
        with torch.no_grad():
            model_manager.set_mode_feature()
            _ = model_manager.cnn(frame_rgb)

    stats = cuda_timer(fn, warmup=3, repeats=10)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SNI-SLAM Performance Benchmark')
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    cfg = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    device = cfg['device']
    scale = cfg['scale']

    print("=" * 60)
    print("SNI-SLAM Performance Benchmark")
    print("=" * 60)
    print(f"Config:  {args.config}")
    print(f"Device:  {device} ({torch.cuda.get_device_name()})")
    print(f"Image:   {cfg['cam']['H']}x{cfg['cam']['W']}")
    print(f"Mapping: {cfg['mapping']['pixels']} rays/iter, {cfg['mapping']['iters']} iters/frame")
    print(f"Tracking: {cfg['tracking']['pixels']} rays/iter, {cfg['tracking']['iters']} iters/frame")
    print()

    # ---- Setup ----
    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    bound = torch.from_numpy(np.array(cfg['mapping']['bound']) * scale).float()
    decoders = config.get_model(cfg, bound).to(device)

    from src.networks.model_manager import ModelManager
    model_manager = ModelManager(cfg)

    # Lightweight renderer (no full SNI_SLAM init)
    from src.utils.Renderer import Renderer

    class _Stub:
        pass
    stub = _Stub()
    stub.bound = bound
    stub.device = device
    stub.model_manager = model_manager
    stub.H, stub.W = cfg['cam']['H'], cfg['cam']['W']
    stub.fx, stub.fy = cfg['cam']['fx'], cfg['cam']['fy']
    stub.cx, stub.cy = cfg['cam']['cx'], cfg['cam']['cy']
    renderer = Renderer(cfg, stub)

    frame_reader = get_dataset(cfg, args, scale, device='cpu')
    frame_data = frame_reader[0]  # first frame

    mem_after_init = gpu_mem_mb()

    # ---- Run benchmarks ----
    results = {}

    print("Running benchmarks...\n")

    print("[1/7] Hash field forward (500K points)...")
    results["hash_field_forward"] = bench_hash_field_forward(decoders, bound, device)

    print("[2/7] Full decoder forward (200K points)...")
    results["decoder_forward"] = bench_decoder_forward(decoders, bound, device)

    print("[3/7] Decoder forward+backward (50K points)...")
    results["decoder_backward"] = bench_decoder_backward(decoders, bound, device)

    print("[4/7] Render batch of rays (mapping config)...")
    results["render_batch"] = bench_render_batch(renderer, decoders, frame_data, cfg, device)

    print("[5/7] CNN feature extraction (DINOv2)...")
    results["cnn_feature"] = bench_cnn_feature_extraction(model_manager, frame_data, cfg, device)

    print("[6/7] Full mapping iteration...")
    results["mapping_iteration"] = bench_mapping_iteration(
        decoders, renderer, model_manager, frame_data, cfg, device)

    print("[7/7] Full tracking iteration...")
    results["tracking_iteration"] = bench_tracking_iteration(
        decoders, renderer, frame_data, cfg, device)

    mem_peak = gpu_mem_mb()

    # ---- Derived estimates ----
    map_iter_ms = results["mapping_iteration"]["mean_ms"]
    track_iter_ms = results["tracking_iteration"]["mean_ms"]
    cnn_ms = results["cnn_feature"]["mean_ms"]
    map_iters = cfg['mapping']['iters']
    track_iters = cfg['tracking']['iters']
    every_frame = cfg['mapping']['every_frame']

    # Per-frame cost estimate (not counting first frame's 1000 iters)
    map_per_frame_ms = map_iter_ms * map_iters + cnn_ms
    track_per_frame_ms = track_iter_ms * track_iters
    # Mapping runs every `every_frame` frames, tracking runs every frame
    # They run in parallel, so wall time ≈ max(mapper_time, tracker_time) per frame
    effective_per_frame_ms = max(map_per_frame_ms, track_per_frame_ms * every_frame)

    n_frames = len(frame_reader)
    est_total_min = (effective_per_frame_ms * n_frames / every_frame) / 1000 / 60

    # ---- Print results ----
    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()

    print("Per-stage timings (mean ± std, ms):")
    print("-" * 60)
    for name, stats in results.items():
        extra = ""
        if "throughput_Mpts_s" in stats:
            extra = f"  [{stats['throughput_Mpts_s']} Mpts/s]"
        elif "throughput_Krays_s" in stats:
            extra = f"  [{stats['throughput_Krays_s']} Krays/s]"
        print(f"  {name:30s}  {stats['mean_ms']:8.2f} ± {stats['std_ms']:5.2f}{extra}")

    print()
    print("Estimated per-frame cost:")
    print("-" * 60)
    print(f"  Mapping  ({map_iters} iters + CNN):  {map_per_frame_ms:8.1f} ms")
    print(f"  Tracking ({track_iters} iters):       {track_per_frame_ms:8.1f} ms")
    print(f"  Effective (parallel):          {effective_per_frame_ms:8.1f} ms")
    print(f"  Est. total for {n_frames} frames:    ~{est_total_min:.1f} min")

    print()
    print("Memory:")
    print("-" * 60)
    print(f"  After model init:  {mem_after_init:8.1f} MB")
    print(f"  Peak during bench: {mem_peak:8.1f} MB")

    print()
    print("Model parameters:")
    print("-" * 60)
    total_params = sum(p.numel() for p in decoders.parameters())
    hash_params = sum(p.numel() for p in list(decoders.hash_sdf.parameters())
                      + list(decoders.hash_color.parameters())
                      + list(decoders.hash_semantic.parameters()))
    mlp_params = total_params - hash_params
    print(f"  Hash grids:  {hash_params:>12,} ({hash_params/total_params*100:.1f}%)")
    print(f"  MLPs:        {mlp_params:>12,} ({mlp_params/total_params*100:.1f}%)")
    print(f"  Total:       {total_params:>12,}")

    print()

    # ---- Save to JSON ----
    output_dir = args.output or cfg['data'].get('output', 'output/benchmark')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'benchmark_results.json')

    save_data = {
        "config": args.config,
        "device": torch.cuda.get_device_name(),
        "image_size": f"{cfg['cam']['H']}x{cfg['cam']['W']}",
        "n_frames": n_frames,
        "timings": results,
        "estimates": {
            "mapping_per_frame_ms": map_per_frame_ms,
            "tracking_per_frame_ms": track_per_frame_ms,
            "effective_per_frame_ms": effective_per_frame_ms,
            "est_total_min": est_total_min,
        },
        "memory": {
            "after_init_mb": mem_after_init,
            "peak_mb": mem_peak,
        },
        "params": {
            "hash_grids": hash_params,
            "mlps": mlp_params,
            "total": total_params,
        },
    }

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {out_path}")
    print()


if __name__ == '__main__':
    main()

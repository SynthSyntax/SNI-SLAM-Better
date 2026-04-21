"""
SNI-SLAM Performance Benchmark (Feature Planes)
================================================
Measures speed of each pipeline stage in isolation.
Uses the first 20 frames of a scene. Total runtime: ~2-5 min on GPU.

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
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
    }


def gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def make_planes(cfg, bound, device):
    """Initialise feature planes the same way SNI_SLAM.init_planes does."""
    c_dim    = cfg['model']['c_dim']
    xyz_len  = bound[:, 1] - bound[:, 0]

    def _make_group(res_list):
        xy, xz, yz = [], [], []
        for grid_res in res_list:
            gs = list(map(int, (xyz_len / grid_res).tolist()))
            gs[0], gs[2] = gs[2], gs[0]
            xy.append(torch.empty([1, c_dim, *gs[1:]]).normal_(mean=0, std=0.01).to(device))
            xz.append(torch.empty([1, c_dim, gs[0], gs[2]]).normal_(mean=0, std=0.01).to(device))
            yz.append(torch.empty([1, c_dim, *gs[:2]]).normal_(mean=0, std=0.01).to(device))
        return xy, xz, yz

    sdf_xy,   sdf_xz,   sdf_yz   = _make_group([cfg['planes_res']['coarse'],   cfg['planes_res']['fine']])
    c_xy,     c_xz,     c_yz     = _make_group([cfg['c_planes_res']['coarse'], cfg['c_planes_res']['fine']])
    s_xy,     s_xz,     s_yz     = _make_group([cfg['s_planes_res']['coarse'], cfg['s_planes_res']['fine']])

    all_planes = (sdf_xy, sdf_xz, sdf_yz, c_xy, c_xz, c_yz, s_xy, s_xz, s_yz)
    return all_planes


def count_plane_params(all_planes):
    return sum(p.numel() for group in all_planes for p in group)


# ---------------------------------------------------------------------------
# Benchmark stages
# ---------------------------------------------------------------------------

def bench_plane_lookup(all_planes, bound, device):
    """Raw feature plane bilinear sampling throughput."""
    from src.common import normalize_3d_coordinate, sample_pdf
    n_points = 500_000
    pts = torch.rand(n_points, 3, device=device)
    lo = bound[:, 0].to(device)
    hi = bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)

    sdf_planes = all_planes[:3]

    def fn():
        with torch.no_grad():
            from src.common import normalize_3d_coordinate
            pts_norm = normalize_3d_coordinate(pts.clone(), bound.to(device))
            feats = []
            for planes in sdf_planes:
                for plane in planes:
                    pts_2d = pts_norm[:, :2].unsqueeze(0).unsqueeze(0)
                    f = torch.nn.functional.grid_sample(
                        plane, pts_2d, align_corners=True, mode='bilinear', padding_mode='border')
                    feats.append(f.squeeze())

    stats = cuda_timer(fn)
    stats["n_points"] = n_points
    stats["throughput_Mpts_s"] = round(n_points / stats["mean_ms"] / 1000, 2)
    return stats


def bench_decoder_forward(decoders, all_planes, bound, device):
    """Full decoder forward (planes + MLP)."""
    n_points = 200_000
    pts = torch.rand(n_points, 3, device=device)
    lo = bound[:, 0].to(device)
    hi = bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)

    def fn():
        with torch.no_grad():
            _ = decoders(pts, all_planes)

    stats = cuda_timer(fn)
    stats["n_points"] = n_points
    stats["throughput_Mpts_s"] = round(n_points / stats["mean_ms"] / 1000, 2)
    return stats


def bench_decoder_backward(decoders, all_planes, bound, device):
    """Decoder forward + backward."""
    n_points = 50_000
    pts = torch.rand(n_points, 3, device=device)
    lo = bound[:, 0].to(device)
    hi = bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)

    planes_params = [nn.Parameter(p) for group in all_planes for p in group]
    optimizer = torch.optim.Adam(list(decoders.parameters()) + planes_params, lr=1e-3)

    def fn():
        optimizer.zero_grad()
        raw, _ = decoders(pts, all_planes)
        loss = raw.sum()
        loss.backward()
        optimizer.step()

    stats = cuda_timer(fn, warmup=3, repeats=10)
    stats["n_points"] = n_points
    return stats


def bench_render_batch(renderer, decoders, all_planes, frame_data, cfg, device):
    """Render a batch of rays."""
    _, gt_color, gt_depth, gt_c2w, _ = frame_data
    gt_color  = gt_color.to(device)
    gt_depth  = gt_depth.to(device)
    gt_c2w    = gt_c2w.to(device)

    H, W       = cfg['cam']['H'], cfg['cam']['W']
    fx, fy     = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy     = cfg['cam']['cx'], cfg['cam']['cy']
    truncation = cfg['model']['truncation']
    c_dim      = cfg['model']['c_dim']
    n_rays     = cfg['mapping']['pixels']
    c2w        = gt_c2w.unsqueeze(0)

    def fn():
        with torch.no_grad():
            rays_o, rays_d, batch_depth, batch_color, _, _, _ = get_samples(
                0, H, 0, W, n_rays, H, W, fx, fy, cx, cy,
                c2w, gt_depth.unsqueeze(0), gt_color.unsqueeze(0),
                device=device, dim=c_dim)
            renderer.render_batch_ray(
                all_planes, decoders, rays_d, rays_o, device, truncation, gt_depth=batch_depth)

    stats = cuda_timer(fn, warmup=2, repeats=8)
    stats["n_rays"] = n_rays
    stats["throughput_Krays_s"] = round(n_rays / stats["mean_ms"], 2)
    return stats


def bench_render_full_image(renderer, decoders, all_planes, frame_data, cfg, device):
    """Render a complete H×W image."""
    _, gt_color, gt_depth, gt_c2w, _ = frame_data
    gt_depth  = gt_depth.to(device)
    gt_c2w    = gt_c2w.to(device)
    truncation = cfg['model']['truncation']

    def fn():
        with torch.no_grad():
            renderer.render_img(all_planes, decoders, gt_c2w, truncation, device, gt_depth=gt_depth)

    stats = cuda_timer(fn, warmup=1, repeats=3)
    H, W = cfg['cam']['H'], cfg['cam']['W']
    stats["resolution"]  = f"{H}x{W}"
    stats["total_rays"]  = H * W
    return stats


def bench_mapping_iteration(decoders, all_planes, renderer, model_manager, frame_data, cfg, device):
    """One full mapping optimization iteration."""
    _, gt_color, gt_depth, gt_c2w, gt_semantic = frame_data
    gt_color    = gt_color.to(device)
    gt_depth    = gt_depth.to(device)
    gt_c2w      = gt_c2w.to(device)
    gt_semantic = gt_semantic.to(device)

    H, W       = cfg['cam']['H'], cfg['cam']['W']
    fx, fy     = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy     = cfg['cam']['cx'], cfg['cam']['cy']
    truncation = cfg['model']['truncation']
    c_dim      = cfg['model']['c_dim']
    n_rays     = cfg['mapping']['pixels']
    c2w        = gt_c2w.unsqueeze(0)

    with torch.no_grad():
        frame_rgb = gt_color.permute(2, 0, 1).unsqueeze(0).to(device)
        model_manager.set_mode_feature()
        sem_feat  = model_manager.cnn(frame_rgb)
        rgb_feat  = model_manager.head(sem_feat)
        sem_feat_sq = sem_feat.squeeze(0)
        rgb_feat_sq = rgb_feat.squeeze(0)

    kf_sem   = sem_feat_sq.unsqueeze(0)
    kf_rgb   = rgb_feat_sq.unsqueeze(0)
    model_manager.set_mode_result()
    with torch.no_grad():
        gt_sem_label = model_manager.cnn(frame_rgb)
    kf_gt_label = gt_sem_label.unsqueeze(0)

    planes_params = [nn.Parameter(p.clone()) for group in all_planes for p in group]
    optimizer = torch.optim.Adam(list(decoders.parameters()) + planes_params, lr=1e-3)
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    w_color    = cfg['mapping']['w_color']
    w_depth    = cfg['mapping']['w_depth']
    w_semantic = cfg['mapping']['w_semantic']
    w_feature  = cfg['mapping']['w_feature']
    w_sdf_fs   = cfg['mapping']['w_sdf_fs']

    def fn():
        optimizer.zero_grad()
        rays_o, rays_d, batch_depth, batch_color, batch_sem, batch_rgb, batch_label = get_samples(
            0, H, 0, W, n_rays, H, W, fx, fy, cx, cy,
            c2w, gt_depth.unsqueeze(0), gt_color.unsqueeze(0),
            sem_feats=kf_sem, rgb_feats=kf_rgb, gt_label=kf_gt_label,
            device=device, dim=c_dim)

        depth, color, sdf, z_vals, gt_feat, plane_feat, render_semantic = renderer.render_batch_ray(
            all_planes, decoders, rays_d, rays_o, device, truncation,
            gt_depth=batch_depth, sem_feats=batch_sem, rgb_feats=batch_rgb, return_emb=True)

        depth_mask = (batch_depth > 0)
        front_mask = (z_vals < (batch_depth[:, None] - truncation)).bool()
        fs_loss       = w_sdf_fs * torch.mean(torch.square(sdf[front_mask] - 1.0))
        color_loss    = w_color * torch.square(batch_color - color).mean()
        depth_loss    = w_depth * torch.square(batch_depth[depth_mask] - depth[depth_mask]).mean()
        feature_loss  = w_feature * (gt_feat.detach() - plane_feat.detach()).abs().mean()
        semantic_loss = w_semantic * ce_loss_fn(render_semantic, batch_label)
        loss = fs_loss + color_loss + depth_loss + feature_loss + semantic_loss
        loss.backward()
        optimizer.step()

    stats = cuda_timer(fn, warmup=2, repeats=8)
    stats["n_rays"] = n_rays
    return stats


def bench_tracking_iteration(decoders, all_planes, renderer, frame_data, cfg, device):
    """One tracking optimization iteration."""
    _, gt_color, gt_depth, gt_c2w, _ = frame_data
    gt_color = gt_color.to(device)
    gt_depth = gt_depth.to(device)
    gt_c2w   = gt_c2w.to(device)

    H, W           = cfg['cam']['H'], cfg['cam']['W']
    fx, fy         = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy         = cfg['cam']['cx'], cfg['cam']['cy']
    truncation     = cfg['model']['truncation']
    c_dim          = cfg['model']['c_dim']
    n_rays         = cfg['tracking']['pixels']
    ignore_edge_H  = cfg['tracking']['ignore_edge_H']
    ignore_edge_W  = cfg['tracking']['ignore_edge_W']

    cam_pose = matrix_to_cam_pose(gt_c2w.unsqueeze(0))
    T = torch.nn.Parameter(cam_pose[:, -3:].clone())
    R = torch.nn.Parameter(cam_pose[:, :4].clone())
    optimizer = torch.optim.Adam([
        {'params': [T], 'lr': cfg['tracking']['lr_T'], 'betas': (0.5, 0.999)},
        {'params': [R], 'lr': cfg['tracking']['lr_R'], 'betas': (0.5, 0.999)},
    ])

    frozen_decoders = copy.deepcopy(decoders)
    for p in frozen_decoders.parameters():
        p.requires_grad_(False)

    def fn():
        cam  = torch.cat([R, T], -1)
        c2w  = cam_pose_to_matrix(cam)
        rays_o, rays_d, batch_depth, batch_color, _, _, _ = get_samples(
            ignore_edge_H, H - ignore_edge_H,
            ignore_edge_W, W - ignore_edge_W,
            n_rays, H, W, fx, fy, cx, cy,
            c2w, gt_depth.unsqueeze(0), gt_color.unsqueeze(0),
            device=device, dim=c_dim)

        depth, color, sdf, z_vals, _, _, _ = renderer.render_batch_ray(
            all_planes, frozen_decoders, rays_d, rays_o, device, truncation, gt_depth=batch_depth)

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
    frame_rgb = gt_color.to(device).permute(2, 0, 1).unsqueeze(0)

    def fn():
        with torch.no_grad():
            model_manager.set_mode_feature()
            _ = model_manager.cnn(frame_rgb)

    return cuda_timer(fn, warmup=3, repeats=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SNI-SLAM Performance Benchmark (Feature Planes)')
    parser.add_argument('config', type=str)
    parser.add_argument('--input_folder', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    cfg    = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    device = cfg['device']
    scale  = cfg['scale']

    print("=" * 60)
    print("SNI-SLAM Performance Benchmark (Feature Planes)")
    print("=" * 60)
    print(f"Config:   {args.config}")
    print(f"Device:   {device} ({torch.cuda.get_device_name()})")
    print(f"Image:    {cfg['cam']['H']}x{cfg['cam']['W']}")
    print(f"Mapping:  {cfg['mapping']['pixels']} rays/iter, {cfg['mapping']['iters']} iters/frame")
    print(f"Tracking: {cfg['tracking']['pixels']} rays/iter, {cfg['tracking']['iters']} iters/frame")
    print()

    torch.cuda.reset_peak_memory_stats()

    bound    = torch.from_numpy(np.array(cfg['mapping']['bound']) * scale).float()
    decoders = config.get_model(cfg).to(device)
    decoders.bound = bound.to(device)
    all_planes = make_planes(cfg, bound, device)

    from src.networks.model_manager import ModelManager
    from src.utils.Renderer import Renderer

    model_manager = ModelManager(cfg)

    class _Stub:
        pass
    stub = _Stub()
    stub.bound = bound; stub.device = device; stub.model_manager = model_manager
    stub.H, stub.W   = cfg['cam']['H'], cfg['cam']['W']
    stub.fx, stub.fy = cfg['cam']['fx'], cfg['cam']['fy']
    stub.cx, stub.cy = cfg['cam']['cx'], cfg['cam']['cy']
    renderer = Renderer(cfg, stub)

    class _Args:
        input_folder = None
    frame_reader = get_dataset(cfg, _Args(), scale, device='cpu')
    frame_data   = frame_reader[0]
    mem_after_init = gpu_mem_mb()

    results = {}
    print("Running benchmarks...\n")

    print("[1/7] Plane lookup (500K points)...")
    results["plane_lookup"] = bench_plane_lookup(all_planes, bound, device)

    print("[2/7] Full decoder forward (200K points)...")
    results["decoder_forward"] = bench_decoder_forward(decoders, all_planes, bound, device)

    print("[3/7] Decoder forward+backward (50K points)...")
    results["decoder_backward"] = bench_decoder_backward(decoders, all_planes, bound, device)

    print("[4/7] Render batch of rays...")
    results["render_batch"] = bench_render_batch(renderer, decoders, all_planes, frame_data, cfg, device)

    print("[5/7] CNN feature extraction (DINOv2)...")
    results["cnn_feature"] = bench_cnn_feature_extraction(model_manager, frame_data, cfg, device)

    print("[6/7] Full mapping iteration...")
    results["mapping_iteration"] = bench_mapping_iteration(decoders, all_planes, renderer, model_manager, frame_data, cfg, device)

    print("[7/7] Full tracking iteration...")
    results["tracking_iteration"] = bench_tracking_iteration(decoders, all_planes, renderer, frame_data, cfg, device)

    mem_peak = gpu_mem_mb()

    map_iter_ms  = results["mapping_iteration"]["mean_ms"]
    track_iter_ms = results["tracking_iteration"]["mean_ms"]
    cnn_ms       = results["cnn_feature"]["mean_ms"]
    map_iters    = cfg['mapping']['iters']
    track_iters  = cfg['tracking']['iters']
    every_frame  = cfg['mapping']['every_frame']
    n_frames     = len(frame_reader)

    map_per_frame_ms      = map_iter_ms * map_iters + cnn_ms
    track_per_frame_ms    = track_iter_ms * track_iters
    effective_per_frame_ms = max(map_per_frame_ms, track_per_frame_ms * every_frame)
    est_total_min         = (effective_per_frame_ms * n_frames / every_frame) / 1000 / 60

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\nPer-stage timings (mean ± std, ms):")
    print("-" * 60)
    for name, stats in results.items():
        extra = ""
        if "throughput_Mpts_s" in stats:
            extra = f"  [{stats['throughput_Mpts_s']} Mpts/s]"
        elif "throughput_Krays_s" in stats:
            extra = f"  [{stats['throughput_Krays_s']} Krays/s]"
        print(f"  {name:30s}  {stats['mean_ms']:8.2f} ± {stats['std_ms']:5.2f}{extra}")

    print(f"\nEstimated per-frame cost:")
    print("-" * 60)
    print(f"  Mapping  ({map_iters} iters + CNN):  {map_per_frame_ms:8.1f} ms")
    print(f"  Tracking ({track_iters} iters):       {track_per_frame_ms:8.1f} ms")
    print(f"  Effective (parallel):          {effective_per_frame_ms:8.1f} ms")
    print(f"  Est. total for {n_frames} frames:    ~{est_total_min:.1f} min")

    print(f"\nMemory:")
    print("-" * 60)
    print(f"  After model init:  {mem_after_init:8.1f} MB")
    print(f"  Peak during bench: {mem_peak:8.1f} MB")

    decoder_params = sum(p.numel() for p in decoders.parameters())
    plane_params   = count_plane_params(all_planes)
    total_params   = decoder_params + plane_params
    print(f"\nModel parameters:")
    print("-" * 60)
    print(f"  Feature planes: {plane_params:>12,} ({plane_params/total_params*100:.1f}%)")
    print(f"  MLPs:           {decoder_params:>12,} ({decoder_params/total_params*100:.1f}%)")
    print(f"  Total:          {total_params:>12,}")
    print()

    output_dir = args.output or cfg['data'].get('output', 'output/benchmark')
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'benchmark_results.json')

    with open(out_path, 'w') as f:
        json.dump({
            "config": args.config,
            "device": torch.cuda.get_device_name(),
            "image_size": f"{cfg['cam']['H']}x{cfg['cam']['W']}",
            "n_frames": n_frames,
            "timings": results,
            "estimates": {
                "mapping_per_frame_ms":   map_per_frame_ms,
                "tracking_per_frame_ms":  track_per_frame_ms,
                "effective_per_frame_ms": effective_per_frame_ms,
                "est_total_min":          est_total_min,
            },
            "memory": {
                "after_init_mb": mem_after_init,
                "peak_mb":       mem_peak,
            },
            "params": {
                "feature_planes": plane_params,
                "mlps":           decoder_params,
                "total":          total_params,
            },
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()

"""
SNI-SLAM Full Evaluation (Feature Planes)
==========================================
Runs all metrics for a completed run and prints one table.

Timing  : runs benchmark logic on ~20 frames (fast, no full re-run needed)
Quality : reads checkpoint + mesh from the completed run output directory

Usage:
    python evaluate.py configs/Replica/room1.yaml
    python evaluate.py configs/Replica/room1.yaml --output output/Replica/room1/test
    python evaluate.py configs/Replica/room1.yaml --gt_mesh /path/to/gt_mesh.ply
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.append('.')
from src import config
from src.common import get_samples, matrix_to_cam_pose, cam_pose_to_matrix
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
    c_dim   = cfg['model']['c_dim']
    xyz_len = bound[:, 1] - bound[:, 0]

    def _make_group(res_list):
        xy, xz, yz = [], [], []
        for grid_res in res_list:
            gs = list(map(int, (xyz_len / grid_res).tolist()))
            gs[0], gs[2] = gs[2], gs[0]
            xy.append(torch.empty([1, c_dim, *gs[1:]]).normal_(mean=0, std=0.01).to(device))
            xz.append(torch.empty([1, c_dim, gs[0], gs[2]]).normal_(mean=0, std=0.01).to(device))
            yz.append(torch.empty([1, c_dim, *gs[:2]]).normal_(mean=0, std=0.01).to(device))
        return xy, xz, yz

    sdf_xy, sdf_xz, sdf_yz = _make_group([cfg['planes_res']['coarse'],   cfg['planes_res']['fine']])
    c_xy,   c_xz,   c_yz   = _make_group([cfg['c_planes_res']['coarse'], cfg['c_planes_res']['fine']])
    s_xy,   s_xz,   s_yz   = _make_group([cfg['s_planes_res']['coarse'], cfg['s_planes_res']['fine']])
    return (sdf_xy, sdf_xz, sdf_yz, c_xy, c_xz, c_yz, s_xy, s_xz, s_yz)


def count_plane_params(all_planes):
    return sum(p.numel() for group in all_planes for p in group)


# ---------------------------------------------------------------------------
# Benchmark stages
# ---------------------------------------------------------------------------

def bench_plane_lookup(all_planes, bound, device):
    n_points = 500_000
    pts = torch.rand(n_points, 3, device=device)
    lo, hi = bound[:, 0].to(device), bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)
    sdf_planes = all_planes[:3]

    def fn():
        with torch.no_grad():
            from src.common import normalize_3d_coordinate
            pts_norm = normalize_3d_coordinate(pts.clone(), bound.to(device))
            for planes in sdf_planes:
                for plane in planes:
                    pts_2d = pts_norm[:, :2].unsqueeze(0).unsqueeze(0)
                    torch.nn.functional.grid_sample(
                        plane, pts_2d, align_corners=True, mode='bilinear', padding_mode='border')

    stats = cuda_timer(fn)
    stats["n_points"] = n_points
    stats["throughput_Mpts_s"] = round(n_points / stats["mean_ms"] / 1000, 2)
    return stats


def bench_decoder_forward(decoders, all_planes, bound, device):
    n_points = 200_000
    pts = torch.rand(n_points, 3, device=device)
    lo, hi = bound[:, 0].to(device), bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)

    def fn():
        with torch.no_grad():
            _ = decoders(pts, all_planes)

    stats = cuda_timer(fn)
    stats["n_points"] = n_points
    stats["throughput_Mpts_s"] = round(n_points / stats["mean_ms"] / 1000, 2)
    return stats


def bench_decoder_backward(decoders, all_planes, bound, device):
    n_points = 50_000
    pts = torch.rand(n_points, 3, device=device)
    lo, hi = bound[:, 0].to(device), bound[:, 1].to(device)
    pts = lo + pts * (hi - lo)
    planes_params = [nn.Parameter(p) for group in all_planes for p in group]
    optimizer = torch.optim.Adam(list(decoders.parameters()) + planes_params, lr=1e-3)

    def fn():
        optimizer.zero_grad()
        raw, _ = decoders(pts, all_planes)
        raw.sum().backward()
        optimizer.step()

    stats = cuda_timer(fn, warmup=3, repeats=10)
    stats["n_points"] = n_points
    return stats


def bench_render_batch(renderer, decoders, all_planes, frame_data, cfg, device):
    _, gt_color, gt_depth, gt_c2w, _ = frame_data
    gt_color = gt_color.to(device)
    gt_depth = gt_depth.to(device)
    gt_c2w   = gt_c2w.to(device)

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


def bench_cnn_feature_extraction(model_manager, frame_data, cfg, device):
    _, gt_color, _, _, _ = frame_data
    frame_rgb = gt_color.to(device).permute(2, 0, 1).unsqueeze(0)

    def fn():
        with torch.no_grad():
            model_manager.set_mode_feature()
            _ = model_manager.cnn(frame_rgb)

    return cuda_timer(fn, warmup=3, repeats=10)


def bench_mapping_iteration(decoders, all_planes, renderer, model_manager, frame_data, cfg, device):
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
        sem_feat    = model_manager.cnn(frame_rgb)
        rgb_feat    = model_manager.head(sem_feat)
        sem_feat_sq = sem_feat.squeeze(0)
        rgb_feat_sq = rgb_feat.squeeze(0)

    kf_sem = sem_feat_sq.unsqueeze(0)
    kf_rgb = rgb_feat_sq.unsqueeze(0)
    model_manager.set_mode_result()
    with torch.no_grad():
        gt_sem_label = model_manager.cnn(frame_rgb)
    kf_gt_label = gt_sem_label.unsqueeze(0)

    planes_params = [nn.Parameter(p.clone()) for group in all_planes for p in group]
    optimizer  = torch.optim.Adam(list(decoders.parameters()) + planes_params, lr=1e-3)
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

        front_mask = (z_vals < (batch_depth[:, None] - truncation)).bool()
        depth_mask = (batch_depth > 0)
        loss = (w_sdf_fs   * torch.mean(torch.square(sdf[front_mask] - 1.0))
              + w_color    * torch.square(batch_color - color).mean()
              + w_depth    * torch.square(batch_depth[depth_mask] - depth[depth_mask]).mean()
              + w_feature  * (gt_feat.detach() - plane_feat.detach()).abs().mean()
              + w_semantic * ce_loss_fn(render_semantic, batch_label))
        loss.backward()
        optimizer.step()

    stats = cuda_timer(fn, warmup=2, repeats=8)
    stats["n_rays"] = n_rays
    return stats


def bench_tracking_iteration(decoders, all_planes, renderer, frame_data, cfg, device):
    _, gt_color, gt_depth, gt_c2w, _ = frame_data
    gt_color = gt_color.to(device)
    gt_depth = gt_depth.to(device)
    gt_c2w   = gt_c2w.to(device)

    H, W          = cfg['cam']['H'], cfg['cam']['W']
    fx, fy        = cfg['cam']['fx'], cfg['cam']['fy']
    cx, cy        = cfg['cam']['cx'], cfg['cam']['cy']
    truncation    = cfg['model']['truncation']
    c_dim         = cfg['model']['c_dim']
    n_rays        = cfg['tracking']['pixels']
    ignore_edge_H = cfg['tracking']['ignore_edge_H']
    ignore_edge_W = cfg['tracking']['ignore_edge_W']

    cam_pose = matrix_to_cam_pose(gt_c2w.unsqueeze(0))
    T = torch.nn.Parameter(cam_pose[:, -3:].clone())
    R = torch.nn.Parameter(cam_pose[:, :4].clone())
    optimizer = torch.optim.Adam([
        {'params': [T], 'lr': cfg['tracking']['lr_T'], 'betas': (0.5, 0.999)},
        {'params': [R], 'lr': cfg['tracking']['lr_R'], 'betas': (0.5, 0.999)},
    ])

    frozen = copy.deepcopy(decoders)
    for p in frozen.parameters():
        p.requires_grad_(False)

    def fn():
        cam    = torch.cat([R, T], -1)
        c2w    = cam_pose_to_matrix(cam)
        rays_o, rays_d, batch_depth, batch_color, _, _, _ = get_samples(
            ignore_edge_H, H - ignore_edge_H,
            ignore_edge_W, W - ignore_edge_W,
            n_rays, H, W, fx, fy, cx, cy,
            c2w, gt_depth.unsqueeze(0), gt_color.unsqueeze(0),
            device=device, dim=c_dim)

        depth, color, sdf, z_vals, _, _, _ = renderer.render_batch_ray(
            all_planes, frozen, rays_d, rays_o, device, truncation, gt_depth=batch_depth)

        depth_mask = (batch_depth > 0)
        loss = (torch.square(batch_depth[depth_mask] - depth[depth_mask]).mean()
              + 5.0 * torch.square(batch_color - color).mean())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    stats = cuda_timer(fn, warmup=2, repeats=8)
    stats["n_rays"] = n_rays
    return stats


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def eval_ate(output_dir, scale):
    from src.tools.eval_ate import convert_poses, evaluate_ate

    ckptsdir = os.path.join(output_dir, 'ckpts')
    if not os.path.exists(ckptsdir):
        return None
    ckpts = sorted([f for f in os.listdir(ckptsdir) if f.endswith('.tar')])
    if not ckpts:
        return None

    ckpt      = torch.load(os.path.join(ckptsdir, ckpts[-1]), map_location='cpu')
    N         = ckpt['idx']
    poses_gt,  mask = convert_poses(ckpt['gt_c2w_list'],       N, scale)
    poses_est, _    = convert_poses(ckpt['estimate_c2w_list'], N, scale)
    poses_est = poses_est[mask]

    gt_np  = poses_gt.cpu().numpy()
    est_np = poses_est.cpu().numpy()
    Np     = gt_np.shape[0]
    r = evaluate_ate({i: gt_np[i] for i in range(Np)},
                     {i: est_np[i] for i in range(Np)})
    return {
        'ate_rmse_cm':   r['absolute_translational_error.rmse'],
        'ate_mean_cm':   r['absolute_translational_error.mean'],
        'ate_median_cm': r['absolute_translational_error.median'],
    }


def eval_mesh_3d(output_dir, gt_mesh_path):
    import trimesh
    from src.tools.eval_recon import accuracy, completion, completion_ratio, get_align_transformation

    for name in ['final_mesh_color_culled.ply', 'final_mesh_color.ply']:
        rec_path = os.path.join(output_dir, 'mesh', name)
        if os.path.exists(rec_path):
            break
    else:
        print("  [mesh] No mesh found — skipping 3D metrics")
        return None

    print(f"  [mesh] Loading {os.path.basename(rec_path)} ...")
    mesh_rec = trimesh.load(rec_path, process=False)
    mesh_gt  = trimesh.load(gt_mesh_path, process=False)
    T        = get_align_transformation(rec_path, gt_mesh_path)
    mesh_rec = mesh_rec.apply_transform(T)

    n      = 450000
    rec_pc = trimesh.sample.sample_surface(mesh_rec, n)[0]
    gt_pc  = trimesh.sample.sample_surface(mesh_gt,  n)[0]

    return {
        'accuracy_cm':        accuracy(gt_pc, rec_pc) * 100,
        'completion_cm':      completion(gt_pc, rec_pc) * 100,
        'completion_ratio_%': completion_ratio(gt_pc, rec_pc) * 100,
    }


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def fmt(v, decimals=3, unit=''):
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{decimals}f}{unit}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def section(title, rows):
    print(f"\n  {'─'*66}")
    print(f"  {title}")
    print(f"  {'─'*66}")
    for label, value, unit, note in rows:
        v = fmt(value, unit=unit)
        print(f"  {label:<40} {v:>12}   {note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--output',   type=str, default=None)
    parser.add_argument('--gt_mesh',  type=str, default=None)
    parser.add_argument('--skip_seg', action='store_true')
    args = parser.parse_args()

    cfg    = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    device = cfg['device']
    scale  = cfg['scale']
    output = args.output or cfg['data']['output']

    print()
    print("=" * 70)
    print("  SNI-SLAM Evaluation (Feature Planes)")
    print("=" * 70)
    print(f"  Config : {args.config}")
    print(f"  Output : {output}")
    print(f"  Device : {device} ({torch.cuda.get_device_name()})")

    results = {}

    # ── 1. Timing ─────────────────────────────────────────────────────────
    print("\n  Running benchmark timing (~2-5 min)...")
    torch.cuda.reset_peak_memory_stats()

    bound      = torch.from_numpy(np.array(cfg['mapping']['bound']) * scale).float()
    decoders   = config.get_model(cfg).to(device)
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

    timing = {}
    timing['plane_lookup']  = bench_plane_lookup(all_planes, bound, device)
    timing['decoder_fwd']   = bench_decoder_forward(decoders, all_planes, bound, device)
    timing['decoder_bwd']   = bench_decoder_backward(decoders, all_planes, bound, device)
    timing['render_batch']  = bench_render_batch(renderer, decoders, all_planes, frame_data, cfg, device)
    timing['cnn']           = bench_cnn_feature_extraction(model_manager, frame_data, cfg, device)
    timing['mapping_iter']  = bench_mapping_iteration(decoders, all_planes, renderer, model_manager, frame_data, cfg, device)
    timing['tracking_iter'] = bench_tracking_iteration(decoders, all_planes, renderer, frame_data, cfg, device)
    mem_peak = gpu_mem_mb()

    map_iters    = cfg['mapping']['iters']
    track_iters  = cfg['tracking']['iters']
    every_frame  = cfg['mapping']['every_frame']
    n_frames     = len(frame_reader)
    map_ms       = timing['mapping_iter']['mean_ms'] * map_iters + timing['cnn']['mean_ms']
    track_ms     = timing['tracking_iter']['mean_ms'] * track_iters
    effective_ms = max(map_ms, track_ms * every_frame)
    est_min      = effective_ms * n_frames / every_frame / 1000 / 60

    results['timing']    = timing
    results['estimates'] = {
        'mapping_per_frame_ms':   map_ms,
        'tracking_per_frame_ms':  track_ms,
        'effective_per_frame_ms': effective_ms,
        'est_total_min':          est_min,
    }
    results['memory'] = {'after_init_mb': mem_after_init, 'peak_mb': mem_peak}

    decoder_params = sum(p.numel() for p in decoders.parameters())
    plane_params   = count_plane_params(all_planes)
    total_params   = decoder_params + plane_params
    results['params'] = {'feature_planes': plane_params, 'mlps': decoder_params, 'total': total_params}

    del decoders, all_planes

    # ── 2. ATE ────────────────────────────────────────────────────────────
    print("  Computing ATE...")
    ate = eval_ate(output, scale)
    results['ate'] = ate

    # ── 3. 3D mesh ────────────────────────────────────────────────────────
    mesh_metrics = None
    if args.gt_mesh:
        print("  Computing 3D mesh metrics...")
        mesh_metrics = eval_mesh_3d(output, args.gt_mesh)
        results['mesh'] = mesh_metrics
    else:
        print("  [mesh] Skipped (no --gt_mesh provided)")

    # ── 4. Print table ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    section("Speed — per-stage (mean ms, lower is better)", [
        ("Plane lookup (500K pts)",          timing['plane_lookup']['mean_ms'],  " ms", f"± {timing['plane_lookup']['std_ms']:.2f}  |  {timing['plane_lookup']['throughput_Mpts_s']} Mpts/s"),
        ("Full decoder forward (200K pts)",  timing['decoder_fwd']['mean_ms'],   " ms", f"± {timing['decoder_fwd']['std_ms']:.2f}  |  {timing['decoder_fwd']['throughput_Mpts_s']} Mpts/s"),
        ("Decoder forward+backward",         timing['decoder_bwd']['mean_ms'],   " ms", f"± {timing['decoder_bwd']['std_ms']:.2f}"),
        ("Render ray batch",                 timing['render_batch']['mean_ms'],  " ms", f"± {timing['render_batch']['std_ms']:.2f}  |  {timing['render_batch']['throughput_Krays_s']} Krays/s"),
        ("CNN feature extraction",           timing['cnn']['mean_ms'],           " ms", f"± {timing['cnn']['std_ms']:.2f}"),
        ("Mapping iteration",                timing['mapping_iter']['mean_ms'],  " ms", f"± {timing['mapping_iter']['std_ms']:.2f}"),
        ("Tracking iteration",               timing['tracking_iter']['mean_ms'], " ms", f"± {timing['tracking_iter']['std_ms']:.2f}"),
    ])

    e = results['estimates']
    section("Speed — estimated per-frame cost", [
        (f"Mapping  ({map_iters} iters + CNN)",  e['mapping_per_frame_ms'],   " ms", ""),
        (f"Tracking ({track_iters} iters)",       e['tracking_per_frame_ms'],  " ms", ""),
        ("Effective (parallel)",                  e['effective_per_frame_ms'], " ms", ""),
        (f"Est. total ({n_frames} frames)",        e['est_total_min'],          " min", ""),
    ])

    section("Memory", [
        ("After model init",  mem_after_init, " MB", ""),
        ("Peak during bench", mem_peak,       " MB", ""),
    ])

    section("Model parameters", [
        ("Feature planes", plane_params,   "", f"({plane_params/total_params*100:.1f}%)"),
        ("MLPs",           decoder_params, "", f"({decoder_params/total_params*100:.1f}%)"),
        ("Total",          total_params,   "", ""),
    ])

    if ate:
        section("Tracking quality (lower is better)", [
            ("ATE RMSE (cm)",   ate['ate_rmse_cm'],   " cm", ""),
            ("ATE mean (cm)",   ate['ate_mean_cm'],   " cm", ""),
            ("ATE median (cm)", ate['ate_median_cm'], " cm", ""),
        ])
    else:
        section("Tracking quality", [("ATE", None, "", "no checkpoint found")])

    if mesh_metrics:
        section("Reconstruction quality", [
            ("Accuracy (cm)",        mesh_metrics['accuracy_cm'],        " cm", "lower is better"),
            ("Completion (cm)",      mesh_metrics['completion_cm'],      " cm", "lower is better"),
            ("Completion Ratio (%)", mesh_metrics['completion_ratio_%'], "%",   "higher is better"),
        ])
    else:
        section("Reconstruction quality", [("(skipped)", None, "", "provide --gt_mesh")])

    print(f"\n  {'─'*66}")
    print()

    # ── 5. Save JSON ──────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    scene_tag = os.path.basename(args.config).replace('.yaml', '')
    out_path  = f'results/eval_{scene_tag}_planes.json'

    def _serial(obj):
        if isinstance(obj, dict):    return {k: _serial(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(_serial(results), f, indent=2)
    print(f"  Saved to {out_path}")
    print()


if __name__ == '__main__':
    main()

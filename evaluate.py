"""
SNI-SLAM Full Evaluation
========================
Runs all metrics for a completed run and prints one table.

Timing  : runs benchmark.py logic on ~20 frames (fast, no full re-run needed)
Quality : reads checkpoint + mesh from the completed run output directory

Usage:
    python evaluate.py configs/Replica/room1.yaml
    python evaluate.py configs/Replica/room1.yaml --output output/Replica/room1/test
    python evaluate.py configs/Replica/room1.yaml --gt_mesh /path/to/gt_mesh.ply

Output:
    Prints a table and saves results/eval_<scene>.json
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.append('.')
from src import config
from src.common import get_samples, get_rays, matrix_to_cam_pose, cam_pose_to_matrix
from src.utils.datasets import get_dataset

# reuse timing helpers from benchmark.py
from benchmark import (
    cuda_timer, gpu_mem_mb,
    bench_hash_field_forward,
    bench_decoder_forward,
    bench_decoder_backward,
    bench_render_batch,
    bench_cnn_feature_extraction,
    bench_mapping_iteration,
    bench_tracking_iteration,
)


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

    ckpt = torch.load(os.path.join(ckptsdir, ckpts[-1]), map_location='cpu')
    N = ckpt['idx']
    poses_gt,  mask = convert_poses(ckpt['gt_c2w_list'],       N, scale)
    poses_est, _    = convert_poses(ckpt['estimate_c2w_list'], N, scale)
    poses_est = poses_est[mask]

    gt_np  = poses_gt.cpu().numpy()
    est_np = poses_est.cpu().numpy()
    Np = gt_np.shape[0]
    r = evaluate_ate({i: gt_np[i] for i in range(Np)},
                     {i: est_np[i] for i in range(Np)})
    return {
        'ate_rmse_cm':   r['absolute_translational_error.rmse'],
        'ate_mean_cm':   r['absolute_translational_error.mean'],
        'ate_median_cm': r['absolute_translational_error.median'],
    }


def eval_mesh_3d(output_dir, gt_mesh_path):
    import trimesh
    from src.tools.eval_recon import (
        accuracy, completion, completion_ratio, get_align_transformation
    )

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
    T = get_align_transformation(rec_path, gt_mesh_path)
    mesh_rec = mesh_rec.apply_transform(T)

    n = 450000
    rec_pc = trimesh.sample.sample_surface(mesh_rec, n)[0]
    gt_pc  = trimesh.sample.sample_surface(mesh_gt,  n)[0]

    return {
        'accuracy_cm':        accuracy(gt_pc, rec_pc) * 100,
        'completion_cm':      completion(gt_pc, rec_pc) * 100,
        'completion_ratio_%': completion_ratio(gt_pc, rec_pc) * 100,
    }


def eval_segmentation_psnr(output_dir, cfg, device):
    """Re-render frame 0 from checkpoint to get PSNR and mIoU."""
    from src.tools.eval_segmentation import Eval_Segmentation
    from src.utils.datasets import get_dataset

    ckptsdir = os.path.join(output_dir, 'ckpts')
    if not os.path.exists(ckptsdir):
        return None
    ckpts = sorted([f for f in os.listdir(ckptsdir) if f.endswith('.tar')])
    if not ckpts:
        return None

    bound_np = np.array(cfg['mapping']['bound']) * cfg['scale']
    bound = torch.from_numpy(bound_np).float().to(device)

    model = config.get_model(cfg, bound.cpu())
    ckpt = torch.load(os.path.join(ckptsdir, ckpts[-1]), map_location=device)
    model.load_state_dict(ckpt['decoder_state_dict'])
    model.to(device)

    class _Args:
        input_folder = None
    frame_reader = get_dataset(cfg, _Args(), cfg['scale'], device=device)

    evaluator = Eval_Segmentation(cfg, device=device)
    truncation = cfg['model']['truncation']

    psnr_list, miou_list = [], []

    # evaluate on first 10 frames (or fewer if short sequence)
    n_eval = min(10, len(frame_reader))
    for i in range(n_eval):
        _, gt_color, gt_depth, gt_c2w, gt_semantic = frame_reader[i]
        gt_depth    = gt_depth.to(device)
        gt_c2w      = gt_c2w.to(device)
        gt_semantic = gt_semantic.to(device)

        semantic, rgb = evaluator.render_img(model, gt_c2w, truncation, device, gt_depth=gt_depth)

        # PSNR
        mse = ((rgb.cpu() - gt_color) ** 2).mean()
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        psnr_list.append(psnr.item())

        # mIoU
        from src.tools.segmentationMetric import SegmentationMetric
        metric = SegmentationMetric(cfg['model']['cnn']['n_classes'])
        pred = semantic.detach().permute(2, 0, 1).unsqueeze(0)
        n_cls = cfg['model']['cnn']['n_classes']
        gt_oh = torch.zeros(1, n_cls, *gt_semantic.shape[-2:], device='cpu')
        gt_idx = gt_semantic.squeeze().cpu().long()
        for c in range(n_cls):
            gt_oh[0, c] = (gt_idx == c).float()
        metric.update(pred.cpu(), gt_oh)
        _, miou = metric.get()
        miou_list.append(miou)

    return {
        'psnr_db':  float(np.mean(psnr_list)),
        'miou_pct': float(np.mean(miou_list) * 100),
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
    parser = argparse.ArgumentParser(description='SNI-SLAM Full Evaluation')
    parser.add_argument('config', type=str)
    parser.add_argument('--output',   type=str, default=None,
                        help='Override output dir from config')
    parser.add_argument('--gt_mesh',  type=str, default=None,
                        help='Ground truth mesh .ply for reconstruction metrics')
    parser.add_argument('--skip_seg', action='store_true',
                        help='Skip PSNR/mIoU (slow re-render)')
    args = parser.parse_args()

    cfg    = config.load_config(args.config, 'configs/SNI-SLAM.yaml')
    device = cfg['device']
    scale  = cfg['scale']
    output = args.output or cfg['data']['output']

    print()
    print("=" * 70)
    print("  SNI-SLAM Evaluation")
    print("=" * 70)
    print(f"  Config : {args.config}")
    print(f"  Output : {output}")
    print(f"  Device : {device} ({torch.cuda.get_device_name()})")

    results = {}

    # ── 1. Timing (benchmark on live model, ~2-5 min) ─────────────────────
    print("\n  Running benchmark timing (~2-5 min)...")
    torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_mb()

    bound    = torch.from_numpy(np.array(cfg['mapping']['bound']) * scale).float()
    decoders = config.get_model(cfg, bound).to(device)

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

    class _Args2:
        input_folder = None
    frame_reader_bench = get_dataset(cfg, _Args2(), scale, device='cpu')
    frame_data = frame_reader_bench[0]
    mem_after_init = gpu_mem_mb()

    timing = {}
    timing['hash_field_fwd']   = bench_hash_field_forward(decoders, bound, device)
    timing['decoder_fwd']      = bench_decoder_forward(decoders, bound, device)
    timing['decoder_bwd']      = bench_decoder_backward(decoders, bound, device)
    timing['render_batch']     = bench_render_batch(renderer, decoders, frame_data, cfg, device)
    timing['cnn']              = bench_cnn_feature_extraction(model_manager, frame_data, cfg, device)
    timing['mapping_iter']     = bench_mapping_iteration(decoders, renderer, model_manager, frame_data, cfg, device)
    timing['tracking_iter']    = bench_tracking_iteration(decoders, renderer, frame_data, cfg, device)
    mem_peak = gpu_mem_mb()

    map_iters    = cfg['mapping']['iters']
    track_iters  = cfg['tracking']['iters']
    every_frame  = cfg['mapping']['every_frame']
    n_frames     = len(frame_reader_bench)
    map_ms       = timing['mapping_iter']['mean_ms'] * map_iters + timing['cnn']['mean_ms']
    track_ms     = timing['tracking_iter']['mean_ms'] * track_iters
    effective_ms = max(map_ms, track_ms * every_frame)
    est_min      = effective_ms * n_frames / every_frame / 1000 / 60

    results['timing'] = timing
    results['estimates'] = {
        'mapping_per_frame_ms':   map_ms,
        'tracking_per_frame_ms':  track_ms,
        'effective_per_frame_ms': effective_ms,
        'est_total_min':          est_min,
    }
    results['memory'] = {'after_init_mb': mem_after_init, 'peak_mb': mem_peak}

    # model size
    total_p = sum(p.numel() for p in decoders.parameters())
    hash_p  = sum(p.numel() for p in (
        list(decoders.hash_sdf.parameters()) +
        list(decoders.hash_color.parameters()) +
        list(decoders.hash_semantic.parameters())))
    mlp_p   = total_p - hash_p
    results['params'] = {'hash_grids': hash_p, 'mlps': mlp_p, 'total': total_p}

    del decoders  # free VRAM before quality eval

    # ── 2. ATE ────────────────────────────────────────────────────────────
    print("  Computing ATE...")
    ate = eval_ate(output, scale)
    results['ate'] = ate

    # ── 3. 3D mesh metrics ────────────────────────────────────────────────
    mesh_metrics = None
    if args.gt_mesh:
        print("  Computing 3D mesh metrics...")
        mesh_metrics = eval_mesh_3d(output, args.gt_mesh)
        results['mesh'] = mesh_metrics
    else:
        print("  [mesh] Skipped (no --gt_mesh provided)")

    # ── 4. PSNR + mIoU ───────────────────────────────────────────────────
    seg_metrics = None
    if not args.skip_seg:
        print("  Computing PSNR + mIoU (re-rendering 10 frames)...")
        try:
            seg_metrics = eval_segmentation_psnr(output, cfg, device)
            results['seg'] = seg_metrics
        except Exception as e:
            print(f"  [seg] Failed: {e}")
    else:
        print("  [seg] Skipped (--skip_seg)")

    # ── 5. Print table ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)

    section("Speed — per-stage (mean ms, lower is better)", [
        ("Hash field forward (500K pts)",  timing['hash_field_fwd']['mean_ms'],  " ms", f"± {timing['hash_field_fwd']['std_ms']:.2f}  |  {timing['hash_field_fwd']['throughput_Mpts_s']} Mpts/s"),
        ("Full decoder forward (200K pts)",timing['decoder_fwd']['mean_ms'],     " ms", f"± {timing['decoder_fwd']['std_ms']:.2f}  |  {timing['decoder_fwd']['throughput_Mpts_s']} Mpts/s"),
        ("Decoder forward+backward",       timing['decoder_bwd']['mean_ms'],     " ms", f"± {timing['decoder_bwd']['std_ms']:.2f}"),
        ("Render ray batch",               timing['render_batch']['mean_ms'],    " ms", f"± {timing['render_batch']['std_ms']:.2f}  |  {timing['render_batch']['throughput_Krays_s']} Krays/s"),
        ("CNN feature extraction",         timing['cnn']['mean_ms'],             " ms", f"± {timing['cnn']['std_ms']:.2f}"),
        ("Mapping iteration",              timing['mapping_iter']['mean_ms'],    " ms", f"± {timing['mapping_iter']['std_ms']:.2f}"),
        ("Tracking iteration",             timing['tracking_iter']['mean_ms'],   " ms", f"± {timing['tracking_iter']['std_ms']:.2f}"),
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
        ("Hash grids", hash_p, "", f"({hash_p/total_p*100:.1f}%)"),
        ("MLPs",       mlp_p,  "", f"({mlp_p/total_p*100:.1f}%)"),
        ("Total",      total_p,"", ""),
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

    if seg_metrics:
        section("Rendering quality", [
            ("PSNR (dB)",  seg_metrics['psnr_db'],  " dB",  "higher is better"),
            ("mIoU (%)",   seg_metrics['miou_pct'], "%",    "higher is better"),
        ])
    else:
        section("Rendering quality", [("(skipped)", None, "", "use --skip_seg to suppress")])

    print(f"\n  {'─'*66}")
    print()

    # ── 6. Save JSON ──────────────────────────────────────────────────────
    os.makedirs('results', exist_ok=True)
    scene_tag = os.path.basename(args.config).replace('.yaml', '')
    out_path  = f'results/eval_{scene_tag}.json'

    # make results JSON-serialisable
    def _to_serialisable(obj):
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, 'w') as f:
        json.dump(_to_serialisable(results), f, indent=2)
    print(f"  Saved to {out_path}")
    print()


if __name__ == '__main__':
    main()

"""
Extensive quality comparison between two completed SNI-SLAM runs.

Works with both feature-planes (main) and hash-grid (hash_encoding) runs,
since it only reads checkpoints, meshes, and benchmark_results.json.

Usage:
    python eval_compare.py \
        --runs planes:output/Replica/room1/planes hash:output/Replica/room1/hash \
        --configs planes:configs/Replica/room1.yaml hash:configs/Replica/room1.yaml \
        --gt_mesh /path/to/gt_mesh.ply
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append('.')
from src import config
from src.tools.eval_ate import convert_poses, evaluate_ate


def eval_ate(output_dir, scale):
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
    gt_np, est_np = poses_gt.cpu().numpy(), poses_est.cpu().numpy()
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
    from src.tools.eval_recon import accuracy, completion, completion_ratio, get_align_transformation
    for name in ['final_mesh_color_culled.ply', 'final_mesh_color.ply']:
        rec_path = os.path.join(output_dir, 'mesh', name)
        if os.path.exists(rec_path):
            break
    else:
        return None
    mesh_rec = trimesh.load(rec_path, process=False)
    mesh_gt  = trimesh.load(gt_mesh_path, process=False)
    mesh_rec = mesh_rec.apply_transform(get_align_transformation(rec_path, gt_mesh_path))
    n = 450000
    rec_pc = trimesh.sample.sample_surface(mesh_rec, n)[0]
    gt_pc  = trimesh.sample.sample_surface(mesh_gt,  n)[0]
    return {
        'accuracy_cm':        accuracy(gt_pc, rec_pc) * 100,
        'completion_cm':      completion(gt_pc, rec_pc) * 100,
        'completion_ratio_%': completion_ratio(gt_pc, rec_pc) * 100,
    }


def load_bench(output_dir):
    path = os.path.join(output_dir, 'benchmark_results.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, decimals=3):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.{decimals}f}"
    return str(v)


def pct_diff(a, b):
    if a is None or b is None or a == 0: return ""
    d = (b - a) / a * 100
    return f"{'+'if d>0 else ''}{d:.1f}%"


def print_section(title, rows, labels):
    la, lb = labels
    print(f"\n  -- {title} --")
    print(f"  {'Metric':<35} {la:>12} {lb:>12}  {'diff':>8}")
    print("  " + "-" * 70)
    for label, va, vb, decimals, lower_better in rows:
        diff   = pct_diff(va, vb)
        marker = ""
        if va is not None and vb is not None:
            if lower_better: marker = " down" if vb < va else (" up" if vb > va else "  ")
            else:            marker = " up"   if vb > va else (" down" if vb < va else "  ")
        print(f"  {label:<35} {fmt(va, decimals):>12} {fmt(vb, decimals):>12}  {diff:>8}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',    nargs='+', required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--gt_mesh', default=None)
    args = parser.parse_args()

    runs    = {k: v for item in args.runs    for k, v in [item.split(':', 1)]}
    cfg_map = {k: v for item in args.configs for k, v in [item.split(':', 1)]}
    labels  = list(runs.keys())
    if len(labels) != 2:
        print("Exactly 2 runs required."); sys.exit(1)
    la, lb = labels
    cfgs = {k: config.load_config(v, 'configs/SNI-SLAM.yaml') for k, v in cfg_map.items()}

    print()
    print("=" * 80)
    print("  SNI-SLAM Comparison")
    print("=" * 80)
    print(f"  {la}: {runs[la]}")
    print(f"  {lb}: {runs[lb]}")

    ate_a = eval_ate(runs[la], cfgs[la]['scale'])
    ate_b = eval_ate(runs[lb], cfgs[lb]['scale'])
    print_section("Tracking -- ATE (lower is better)", [
        ("ATE RMSE (cm)",   ate_a and ate_a['ate_rmse_cm'],   ate_b and ate_b['ate_rmse_cm'],   3, True),
        ("ATE mean (cm)",   ate_a and ate_a['ate_mean_cm'],   ate_b and ate_b['ate_mean_cm'],   3, True),
        ("ATE median (cm)", ate_a and ate_a['ate_median_cm'], ate_b and ate_b['ate_median_cm'], 3, True),
    ], (la, lb))

    if args.gt_mesh:
        m_a = eval_mesh_3d(runs[la], args.gt_mesh)
        m_b = eval_mesh_3d(runs[lb], args.gt_mesh)
        print_section("Reconstruction (accuracy/completion lower better, ratio higher better)", [
            ("Accuracy (cm)",        m_a and m_a['accuracy_cm'],        m_b and m_b['accuracy_cm'],        3, True),
            ("Completion (cm)",      m_a and m_a['completion_cm'],      m_b and m_b['completion_cm'],      3, True),
            ("Completion Ratio (%)", m_a and m_a['completion_ratio_%'], m_b and m_b['completion_ratio_%'], 2, False),
        ], (la, lb))
    else:
        print("\n  -- Reconstruction -- (skipped: no --gt_mesh)")

    ba = load_bench(runs[la])
    bb = load_bench(runs[lb])
    if ba and bb:
        ea, eb = ba['estimates'], bb['estimates']
        ma, mb = ba['memory'],    bb['memory']
        pa, pb = ba['params'],    bb['params']
        print_section("Speed (lower is better)", [
            ("Mapping per frame (ms)",   ea['mapping_per_frame_ms'],   eb['mapping_per_frame_ms'],   1, True),
            ("Tracking per frame (ms)",  ea['tracking_per_frame_ms'],  eb['tracking_per_frame_ms'],  1, True),
            ("Effective per frame (ms)", ea['effective_per_frame_ms'], eb['effective_per_frame_ms'], 1, True),
            ("Est. total run (min)",     ea['est_total_min'],           eb['est_total_min'],           1, True),
        ], (la, lb))
        print_section("Memory & Model size", [
            ("Peak GPU memory (MB)", ma['peak_mb'],  mb['peak_mb'],  1, True),
            ("Encoder params",       pa['total'] - pa.get('mlps', 0), pb['total'] - pb.get('mlps', 0), 0, False),
            ("MLP params",           pa.get('mlps', 0), pb.get('mlps', 0), 0, False),
            ("Total params",         pa['total'],   pb['total'],    0, False),
        ], (la, lb))
    else:
        print("\n  -- Speed -- (run evaluate.py or benchmark.py on each output dir first)")

    print()
    print("  down = better for lower-is-better   up = better for higher-is-better")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()

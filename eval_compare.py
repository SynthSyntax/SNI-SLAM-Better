"""
Extensive quality comparison between two or more completed SNI-SLAM runs.

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


def load_eval(output_dir, label):
    """Load results/eval_*.json produced by evaluate.py.

    Search order:
      1. results/eval_<label>.json          (exact label match)
      2. any results/eval_*.json containing the label in its stem
      3. results/eval_<scene>.json          (scene = second-to-last path component)
    """
    results_dir = 'results'
    if not os.path.isdir(results_dir):
        return None

    # exact match
    exact = os.path.join(results_dir, f'eval_{label}.json')
    if os.path.exists(exact):
        with open(exact) as f:
            return json.load(f)

    # scan for any file whose stem contains the label
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith('.json') and fname.startswith('eval_') and label in fname:
            with open(os.path.join(results_dir, fname)) as f:
                return json.load(f)

    # fall back to scene name derived from path
    scene = os.path.basename(os.path.dirname(output_dir.rstrip('/')))
    scene_path = os.path.join(results_dir, f'eval_{scene}.json')
    if os.path.exists(scene_path):
        with open(scene_path) as f:
            return json.load(f)

    return None


METRIC_W = 36
VAL_W    = 11
DIFF_W   = 11   # 8 for number + space + 2 for arrow symbol


def fmt(v, decimals=3):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.{decimals}f}"
    return str(v)


def pct_diff(base, v):
    if base is None or v is None or base == 0: return ""
    d = (v - base) / base * 100
    return f"{'+'if d>0 else ''}{d:.1f}%"


def print_section(title, rows, labels):
    """rows: list of (metric_label, [val_per_run], decimals, lower_better)"""
    n = len(labels)

    hdr = f"  {'Metric':<{METRIC_W}}"
    for lbl in labels:
        hdr += f" {lbl:>{VAL_W}}"
    for lbl in labels[1:]:
        hdr += f" {'Δ'+lbl:>{DIFF_W}}"

    sep = 2 + METRIC_W + n * (VAL_W + 1) + (n - 1) * (DIFF_W + 1)

    print(f"\n  ── {title} ──")
    print(hdr)
    print("  " + "─" * (sep - 2))

    for metric_label, vals, decimals, lower_better in rows:
        line = f"  {metric_label:<{METRIC_W}}"
        for v in vals:
            line += f" {fmt(v, decimals):>{VAL_W}}"
        base = vals[0]
        for v in vals[1:]:
            diff = pct_diff(base, v)
            arrow = ""
            if base is not None and v is not None and diff:
                if lower_better: arrow = " ↓" if v < base else " ↑"
                else:            arrow = " ↑" if v > base else " ↓"
            cell = f"{diff}{arrow}"
            line += f" {cell:>{DIFF_W}}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs',    nargs='+', required=True)
    parser.add_argument('--configs', nargs='+', default=[],
                        help='Optional: label:config.yaml pairs to get scale. Defaults to scale=1.')
    parser.add_argument('--gt_mesh', default=None)
    args = parser.parse_args()

    runs    = {k: v for item in args.runs    for k, v in [item.split(':', 1)]}
    cfg_map = {k: v for item in args.configs for k, v in [item.split(':', 1)]}
    labels  = list(runs.keys())
    if len(labels) < 2:
        print("At least 2 runs required."); sys.exit(1)

    def get_scale(label):
        if label in cfg_map:
            return config.load_config(cfg_map[label], 'configs/SNI-SLAM.yaml')['scale']
        return 1.0

    print()
    print("=" * 80)
    print("  SNI-SLAM Comparison")
    print("=" * 80)
    for lbl in labels:
        marker = "  [baseline]" if lbl == labels[0] else ""
        print(f"  {lbl}: {runs[lbl]}{marker}")

    ates = [eval_ate(runs[lbl], get_scale(lbl)) for lbl in labels]
    print_section("Tracking -- ATE (lower is better)", [
        ("ATE RMSE (cm)",   [a and a['ate_rmse_cm']   for a in ates], 3, True),
        ("ATE mean (cm)",   [a and a['ate_mean_cm']   for a in ates], 3, True),
        ("ATE median (cm)", [a and a['ate_median_cm'] for a in ates], 3, True),
    ], labels)

    if args.gt_mesh:
        meshes = [eval_mesh_3d(runs[lbl], args.gt_mesh) for lbl in labels]
        print_section("Reconstruction (accuracy/completion lower better, ratio higher better)", [
            ("Accuracy (cm)",        [m and m['accuracy_cm']        for m in meshes], 3, True),
            ("Completion (cm)",      [m and m['completion_cm']      for m in meshes], 3, True),
            ("Completion Ratio (%)", [m and m['completion_ratio_%'] for m in meshes], 2, False),
        ], labels)
    else:
        print("\n  -- Reconstruction -- (skipped: no --gt_mesh)")

    evals = [load_eval(runs[lbl], lbl) for lbl in labels]
    if all(evals):
        estimates = [e['estimates'] for e in evals]
        memories  = [e['memory']    for e in evals]
        params    = [e['params']    for e in evals]

        def enc_params(p):
            return p.get('hash_grids') or p.get('feature_planes') or 0

        print_section("Speed (lower is better)", [
            ("Mapping per frame (ms)",   [e['mapping_per_frame_ms']   for e in estimates], 1, True),
            ("Tracking per frame (ms)",  [e['tracking_per_frame_ms']  for e in estimates], 1, True),
            ("Effective per frame (ms)", [e['effective_per_frame_ms'] for e in estimates], 1, True),
            ("Est. total run (min)",     [e['est_total_min']          for e in estimates], 1, True),
        ], labels)
        print_section("Memory & Model size", [
            ("Peak GPU memory (MB)",   [m['peak_mb']      for m in memories], 1, True),
            ("Encoder params",         [enc_params(p)    for p in params],   0, False),
            ("MLP params",                        [p.get('mlps', 0) for p in params],   0, False),
            ("Total params",                      [p['total']       for p in params],   0, False),
        ], labels)
    else:
        print("\n  -- Speed -- (run evaluate.py on each output dir first)")

    print()
    print("  ↓ = improvement for lower-is-better   ↑ = improvement for higher-is-better")
    print("  Δ columns are relative to the first run (baseline)")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()

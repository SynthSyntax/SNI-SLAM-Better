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
    parser.add_argument('--configs', nargs='+', default=[],
                        help='Optional: label:config.yaml pairs to get scale. Defaults to scale=1.')
    parser.add_argument('--gt_mesh', default=None)
    args = parser.parse_args()

    runs    = {k: v for item in args.runs    for k, v in [item.split(':', 1)]}
    cfg_map = {k: v for item in args.configs for k, v in [item.split(':', 1)]}
    labels  = list(runs.keys())
    if len(labels) != 2:
        print("Exactly 2 runs required."); sys.exit(1)
    la, lb = labels

    def get_scale(label):
        if label in cfg_map:
            return config.load_config(cfg_map[label], 'configs/SNI-SLAM.yaml')['scale']
        return 1.0

    print()
    print("=" * 80)
    print("  SNI-SLAM Comparison")
    print("=" * 80)
    print(f"  {la}: {runs[la]}")
    print(f"  {lb}: {runs[lb]}")

    ate_a = eval_ate(runs[la], get_scale(la))
    ate_b = eval_ate(runs[lb], get_scale(lb))
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

    ba = load_eval(runs[la], la)
    bb = load_eval(runs[lb], lb)
    if ba and bb:
        ea, eb = ba['estimates'], bb['estimates']
        ma, mb = ba['memory'],    bb['memory']
        pa, pb = ba['params'],    bb['params']
        # encoder key differs: hash uses 'hash_grids', planes uses 'feature_planes'
        enc_a = pa.get('hash_grids') or pa.get('feature_planes') or 0
        enc_b = pb.get('hash_grids') or pb.get('feature_planes') or 0
        enc_label_a = 'hash grids' if 'hash_grids' in pa else 'feature planes'
        enc_label_b = 'hash grids' if 'hash_grids' in pb else 'feature planes'
        print_section("Speed (lower is better)", [
            ("Mapping per frame (ms)",   ea['mapping_per_frame_ms'],   eb['mapping_per_frame_ms'],   1, True),
            ("Tracking per frame (ms)",  ea['tracking_per_frame_ms'],  eb['tracking_per_frame_ms'],  1, True),
            ("Effective per frame (ms)", ea['effective_per_frame_ms'], eb['effective_per_frame_ms'], 1, True),
            ("Est. total run (min)",     ea['est_total_min'],          eb['est_total_min'],          1, True),
        ], (la, lb))
        print_section("Memory & Model size", [
            ("Peak GPU memory (MB)",                  ma['peak_mb'], mb['peak_mb'], 1, True),
            (f"Encoder ({enc_label_a} / {enc_label_b})", enc_a,     enc_b,         0, False),
            ("MLP params",                            pa.get('mlps', 0), pb.get('mlps', 0), 0, False),
            ("Total params",                          pa['total'],  pb['total'],   0, False),
        ], (la, lb))
    else:
        print("\n  -- Speed -- (run evaluate.py on each output dir first)")

    print()
    print("  down = better for lower-is-better   up = better for higher-is-better")
    print("=" * 80)
    print()


if __name__ == '__main__':
    main()

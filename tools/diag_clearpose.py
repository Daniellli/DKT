"""Diagnose ClearPose + DKT + MoGe data on 1 frame.

Answers three questions before any refinement is touched:
  1. Does the label map only mark transparent objects, or all foreground?
     (unique values, >0 coverage, distribution)
  2. What are the GT depth units? (mm vs m, range, valid-pixel ratio)
  3. What is the DKT pred_disp range, and is it 0-1 normalized or not?
     What is MoGe's depth range (metric)?

Run:
    python tools/diag_clearpose.py --scene_idx 0 --n_frames 21
"""

import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.clearpose_dataset import discover_sequence_frames
from tools.refine_moge import _infer_moge


def banner(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def stats(name, arr, mask=None):
    if mask is not None:
        v = arr[mask]
        cov = float(mask.mean()) * 100
    else:
        v = arr.ravel()
        cov = 100.0
    if v.size == 0:
        print(f"  {name:18s}  no values")
        return
    print(f"  {name:18s}  coverage={cov:5.2f}%  "
          f"min={v.min():>12.4f}  p1={np.percentile(v, 1):>12.4f}  "
          f"p50={np.percentile(v, 50):>12.4f}  p99={np.percentile(v, 99):>12.4f}  "
          f"max={v.max():>12.4f}  dtype={arr.dtype}")


def run_dkt(frames):
    """Run DKT on a short frame list (>=21) and return raw depth_map."""
    import dkt.pipelines.pipelines as dkt_pipe_mod
    from dkt.pipelines.pipelines import DKTPipeline

    pipe = DKTPipeline(is14B=False, is_depth=True)
    orig = dkt_pipe_mod.extract_frames_from_video_file
    dkt_pipe_mod.extract_frames_from_video_file = lambda _p: (frames, 15.0)
    try:
        out = pipe("dummy.mp4", vis_pc=False)
    finally:
        dkt_pipe_mod.extract_frames_from_video_file = orig
    return out["depth_map"]  # [T, H, W] in whatever units DKT produces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clearpose_root", default="/workspace/datasets/clearpose/set2")
    ap.add_argument("--scene_idx", type=int, default=0)
    ap.add_argument("--start", type=int, default=0,
                    help="frame index (after stride) to center the diagnostic on")
    ap.add_argument("--n_frames", type=int, default=21,
                    help="number of frames for DKT (>=21 needed by its sliding window)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--depth_scale", type=float, default=1.0,
                    help="multiply GT by this — 1.0 keeps raw, 0.001 converts mm→m")
    ap.add_argument("--no_dkt", action="store_true",
                    help="skip DKT inference (much faster; only MoGe + raw stats)")
    args = ap.parse_args()

    # ---- pick scene + frames ----
    from tools.clearpose_dataset import discover_sequence_dirs
    scenes = discover_sequence_dirs(args.clearpose_root)
    if not scenes:
        print(f"NO SCENES under {args.clearpose_root}"); return
    scene = scenes[args.scene_idx]
    print(f"scene: {scene}  (idx {args.scene_idx}/{len(scenes)-1})")

    records = discover_sequence_frames(scene)
    records = records[::args.stride]
    if not records:
        print("no records"); return
    n = len(records)
    start = max(0, min(args.start, n - 1))
    end = min(start + args.n_frames, n)
    sel = records[start:end]
    print(f"frames: {start}..{end-1}  (n={len(sel)} of {n}, stride={args.stride})")

    # The "target" frame is the center (or last if n_frames < target)
    target = sel[min(len(sel) - 1, len(sel) // 2)]
    target_idx = start + min(len(sel) - 1, len(sel) // 2)

    # =========================================================
    banner(f"1. LABEL  ({os.path.basename(target.label_path) or 'MISSING'})")
    # =========================================================
    if target.label_path and os.path.exists(target.label_path):
        lbl = cv2.imread(target.label_path, cv2.IMREAD_UNCHANGED)
        print(f"  shape: {lbl.shape}  dtype: {lbl.dtype}")
        if lbl.ndim == 3:
            lbl = lbl[..., 0]
            print("  (took first channel)")
        uniq, counts = np.unique(lbl, return_counts=True)
        total = lbl.size
        print(f"  unique values: {uniq.tolist()}")
        for u, c in zip(uniq.tolist(), counts.tolist()):
            print(f"    label={int(u):>3d}  count={c:>10d}  ({c/total*100:5.2f}%)")
        obj = lbl > 0
        print(f"  >0 (object) coverage: {obj.mean()*100:.2f}% of all pixels")
        # distribution of obj pixel values
        if obj.any():
            obj_vals, obj_counts = np.unique(lbl[obj], return_counts=True)
            top = sorted(zip(obj_counts.tolist(), obj_vals.tolist()),
                         reverse=True)[:8]
            print(f"  top object ids by pixel count: {top}")
    else:
        print("  NO label file found.")

    # =========================================================
    banner(f"2. GT DEPTH  ({os.path.basename(target.depth_path)})")
    # =========================================================
    gt_raw = cv2.imread(target.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(f"  raw shape: {gt_raw.shape}  dtype(source)={gt_raw.dtype}  "
          f"raw min={gt_raw.min():.1f}  raw max={gt_raw.max():.1f}")

    gt = gt_raw * args.depth_scale
    print(f"  after depth_scale={args.depth_scale}:  min={gt.min():.6f}  max={gt.max():.6f}")
    valid = gt > 0
    stats("gt>0 (raw units)", gt_raw, valid)
    if args.depth_scale != 1.0:
        stats(f"gt>0 (×{args.depth_scale})", gt, valid)
    print(f"  → interpretation:")
    if valid.any():
        med = float(np.median(gt_raw[valid]))
        print(f"    median raw GT = {med:.2f}")
        if med > 50:
            print(f"    ⇒ looks like mm (typical desk scene ~500-1500mm)")
        elif med < 5:
            print(f"    ⇒ looks like meters (typical desk scene ~0.5-1.5m)")
        else:
            print(f"    ⇒ ambiguous, neither mm-typical nor m-typical")

    # =========================================================
    banner("3. MoGe DEPTH (metric, m)")
    # =========================================================
    color_pil = Image.open(target.color_path).convert("RGB")
    moge_d, moge_valid = _infer_moge(color_pil, "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  shape: {moge_d.shape}  valid coverage: {moge_valid.mean()*100:.2f}%")
    stats("moge_depth (m)", moge_d, moge_valid)
    if moge_valid.any():
        med = float(np.median(moge_d[moge_valid]))
        print(f"  → median MoGe depth = {med:.3f} m  "
              f"(should be 0.3-2.0m for desk scenes)")

    # =========================================================
    banner("4. DKT pred_disp (raw, before any refinement)")
    # =========================================================
    if args.no_dkt:
        print("  skipped (--no_dkt)")
    else:
        # need a full window for DKT
        win = records[max(0, target_idx - 10): target_idx + 11]
        if len(win) < 21:
            # pad by repeating
            win = (win * (21 // max(1, len(win)) + 1))[:21]
        frames = [Image.open(r.color_path).convert("RGB") for r in win]
        dkt_disp = run_dkt(frames)  # [T, H, W]
        # take the center frame
        center = dkt_disp[len(dkt_disp) // 2]
        print(f"  DKT output shape: {dkt_disp.shape}  "
              f"(T={dkt_disp.shape[0]} H={dkt_disp.shape[1]} W={dkt_disp.shape[2]})")
        pos = center > 0
        stats("dkt_disp center>0", center, pos)
        if pos.any():
            # implied depth = 1/disp
            dkt_depth_implied = np.zeros_like(center)
            dkt_depth_implied[pos] = 1.0 / center[pos]
            stats("implied 1/disp", dkt_depth_implied, pos)
            print(f"  → if dkt_disp is in 1/meters:  median implied depth = "
                  f"{float(np.median(dkt_depth_implied[pos])):.3f} m")
            print(f"  → if dkt_disp is in 1/mm:      median implied depth = "
                  f"{float(np.median(dkt_depth_implied[pos]))*1000:.1f} mm")

    # =========================================================
    banner("5. CROSS-CHECK  GT vs MoGe vs DKT at matching pixels")
    # =========================================================
    if not args.no_dkt and 'moge_d' in dir() and 'center' in dir():
        # resize GT to DKT's resolution
        dkt_h, dkt_w = center.shape
        gt_resized = cv2.resize(gt, (dkt_w, dkt_h), interpolation=cv2.INTER_NEAREST)
        moge_resized = cv2.resize(moge_d, (dkt_w, dkt_h), interpolation=cv2.INTER_NEAREST)
        both_valid = (gt_resized > 0) & (center > 0) & moge_valid
        # MoGe valid is for original size, recompute at DKT res
        moge_valid_resized = cv2.resize(moge_valid.astype(np.uint8),
                                         (dkt_w, dkt_h),
                                         interpolation=cv2.INTER_NEAREST) > 0
        both_valid = (gt_resized > 0) & (center > 0) & moge_valid_resized
        print(f"  pixels where GT>0 AND DKT>0 AND MoGe valid (at DKT res): "
              f"{both_valid.mean()*100:.2f}%")
        if both_valid.any():
            g = gt_resized[both_valid]
            d = 1.0 / center[both_valid]
            m = moge_resized[both_valid]
            print(f"  GT (raw units)        : median={np.median(g):.2f}  "
                  f"p25={np.percentile(g,25):.2f}  p75={np.percentile(g,75):.2f}")
            print(f"  1/DKT_disp  (=depth)  : median={np.median(d):.4f}  "
                  f"p25={np.percentile(d,25):.4f}  p75={np.percentile(d,75):.4f}")
            print(f"  MoGe (m)              : median={np.median(m):.3f}  "
                  f"p25={np.percentile(m,25):.3f}  p75={np.percentile(m,75):.3f}")
            # what scale makes 1/DKT match GT?
            s_to_mm = np.median(g) / np.median(d)
            s_to_m  = np.median(m) / np.median(d)
            print(f"  scale that aligns 1/DKT to GT(median): {s_to_mm:.4f}")
            print(f"  scale that aligns 1/DKT to MoGe:       {s_to_m:.4f}")
            dkt_is_normalized = s_to_mm < 0.01
            dkt_label = "NORMALIZED 0-1 (relative disparity)" if dkt_is_normalized \
                        else "METRIC 1/meters"
            print(f"  => DKT_disp is {dkt_label}")

    # =========================================================
    banner("DONE")
    # =========================================================


if __name__ == "__main__":
    main()

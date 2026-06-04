"""Evaluate DKT on ClearPose with DKT's own protocol.

Units: ClearPose GT is in mm. Default depth_scale=1.0 keeps GT in mm.
RMSE is reported in mm; target DKT 14B paper number: REL~9.72, RMSE~14.58 mm.
"""
import argparse
import os

import numpy as np

from dkt.pipelines.pipelines import DKTPipeline
from tools.clearpose_dataset import (
    DEFAULT_DEPTH_SUFFIXES,
    discover_sequence_dirs,
    discover_sequence_frames,
    load_clearpose_sequence,
)
from tools.eval_utils import transfer_pred_disp2depth


def find_scenes(root, depth_suffix):
    if depth_suffix:
        depth_suffixes = (depth_suffix,) + tuple(s for s in DEFAULT_DEPTH_SUFFIXES if s != depth_suffix)
        return discover_sequence_dirs(root, depth_suffixes=depth_suffixes)
    return discover_sequence_dirs(root)


def load_scene(scene_dir, depth_scale, max_frames, stride, depth_suffix,
               object_mask_suffix="-label.png"):
    import cv2
    frames, gts, masks, records = load_clearpose_sequence(
        scene_dir,
        depth_scale=1.0,
        max_frames=max_frames,
        stride=stride,
        depth_suffix=depth_suffix,
    )
    if not frames:
        return [], None, None, None
    # Load label maps if available (0=background, >0=transparent object)
    labels = []
    for r in records:
        stem = r.color_path[: -len("-color.png")]
        label_path = stem + object_mask_suffix
        if os.path.exists(label_path):
            lbl = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
            labels.append(lbl)
        else:
            labels = None
            break
    if labels is not None:
        labels = np.stack(labels)
    return frames, gts, masks, labels


def compute_metrics(pred_depth, gt_depth, mask):
    p, g = pred_depth[mask], gt_depth[mask]
    eps = 1e-6
    absrel = np.mean(np.abs(p - g) / np.maximum(g, eps)) * 100.0
    rmse = np.sqrt(np.mean((p - g) ** 2))
    ratio = np.maximum(p / np.maximum(g, eps), g / np.maximum(p, eps))
    return (absrel, rmse, np.mean(ratio < 1.05) * 100,
            np.mean(ratio < 1.10) * 100, np.mean(ratio < 1.25) * 100)


from tools.refine_moge import (  # noqa: E402
    moge_depths,
    refine_none,
    refine_moge_scale,
    refine_moge_affine,
    refine_moge_affine_mask,
)
from tools.refine_object_temporal import (  # noqa: E402
    dkt_temporal_moge_anchor,
    moge_affine_mask_median,
    refine_object_temporal,
)

_REFINE_FN = {
    "none":                     refine_none,
    "moge_scale":               refine_moge_scale,
    "moge_affine":              refine_moge_affine,
    "moge_affine_mask":         refine_moge_affine_mask,
    "moge_affine_mask_median":  moge_affine_mask_median,
    "object_temporal":          refine_object_temporal,
    "dkt_temporal_moge_anchor": dkt_temporal_moge_anchor,
}


def run_dkt_on_frames(pipe, frames):
    import dkt.pipelines.pipelines as dkt_pipe_mod

    orig = dkt_pipe_mod.extract_frames_from_video_file
    dkt_pipe_mod.extract_frames_from_video_file = lambda _p: (frames, 15.0)
    try:
        pred = pipe("dummy.mp4", vis_pc=False)
    finally:
        dkt_pipe_mod.extract_frames_from_video_file = orig
    return pred["depth_map"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clearpose_root", required=True)
    ap.add_argument("--depth_suffix", default="-depth_true.png")
    ap.add_argument("--depth_scale", type=float, default=1.0,
                    help="multiply GT depth; default 1.0 keeps mm, RMSE in mm")
    ap.add_argument("--max_depth", type=float, default=10000.0,
                    help="depth upper bound in same units as GT; removes disparity outliers")
    ap.add_argument("--chunk_size", type=int, default=200,
                    help="split long scenes into N-frame chunks to bound memory")
    ap.add_argument("--is14b", action="store_true")
    ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--refine", default="moge_affine_mask",
                    choices=["none", "moge_scale", "moge_affine",
                             "moge_affine_mask", "moge_affine_mask_median",
                             "object_temporal", "dkt_temporal_moge_anchor"],
                    help="refinement strategy (default: moge_affine_mask = current best)")
    ap.add_argument("--temporal_window", type=int, default=11,
                    help="smoothing window for object_temporal refine (frames)")
    ap.add_argument("--median_kernel", type=int, default=5,
                    help="temporal median-filter kernel for moge_affine_mask_median "
                         "and dkt_temporal_moge_anchor")
    ap.add_argument("--object_mask_suffix", default="-label.png",
                    help="filename suffix for transparent-object label files (0=bg, >0=object id)")
    ap.add_argument("--metric_eval", action="store_true",
                    help="metric protocol: skip lstsq re-alignment, use MoGe scale directly; "
                         "also runs MoGe-only baseline for comparison")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    scenes = find_scenes(args.clearpose_root, args.depth_suffix)
    if args.max_scenes:
        scenes = scenes[:args.max_scenes]
    print(f"found {len(scenes)} scenes under {args.clearpose_root}")
    if args.dry_run:
        depth_suffixes = (args.depth_suffix,) + tuple(
            s for s in DEFAULT_DEPTH_SUFFIXES if s != args.depth_suffix
        )
        for s in scenes[:15]:
            frames = discover_sequence_frames(s, depth_suffixes=depth_suffixes)
            mask_count = sum(1 for f in frames if f.mask_path is not None)
            print(f"  {s}: {len(frames)} paired frames, {mask_count} explicit masks")
        return

    pipe = DKTPipeline(is14B=args.is14b, is_depth=True)
    model_label = "DKT-14B" if args.is14b else "DKT-1.3B"
    refine_fn = _REFINE_FN[args.refine]
    print(f"using {model_label}  refine={args.refine}")
    if args.metric_eval:
        print("protocol: METRIC (no lstsq re-alignment; MoGe scale applied directly)")

    all_m, all_m_moge = [], []
    for si, scene in enumerate(scenes):
        frames, gt, mask, labels = load_scene(scene, args.depth_scale, args.max_frames, args.stride, args.depth_suffix, args.object_mask_suffix)
        if not frames:
            print(f"[{si+1}/{len(scenes)}] {scene}: no usable frames, skip")
            continue

        gt = gt * args.depth_scale
        eval_mask = mask & (gt > 1e-3) & (gt < args.max_depth)
        T = len(frames)
        cs = args.chunk_size if args.chunk_size > 0 else T
        if T > cs:
            print(f"WARNING: scene has {T} frames > chunk_size={cs}, inter-chunk scale drift will inflate RMSE")
        if labels is not None:
            print(f"  using ClearPose label masks for opaque-region scale estimation")
        scene_per_frame, scene_per_frame_moge = [], []
        clip_logged = False
        for s in range(0, T, cs):
            e = min(s + cs, T)
            fr_c = frames[s:e]
            gt_c = gt[s:e].astype(np.float32, copy=False)
            eval_mask_c = eval_mask[s:e].astype(bool, copy=False)
            labels_c = labels[s:e] if labels is not None else None

            dkt_disp = run_dkt_on_frames(pipe, fr_c)
            extra_kwargs = {}
            if args.refine == "object_temporal":
                extra_kwargs["temporal_window"] = args.temporal_window
            if args.refine in ("moge_affine_mask_median",
                               "dkt_temporal_moge_anchor"):
                extra_kwargs["median_kernel"] = args.median_kernel
            pred_disp = refine_fn(
                dkt_disp, fr_c,
                label_maps=labels_c,
                **extra_kwargs,
            ).astype(np.float32, copy=False)

            if args.metric_eval:
                # Direct inversion: refined_disp is already in MoGe metric scale (1/m)
                pred_depth = np.where(pred_disp > 1e-6, 1.0 / pred_disp, args.max_depth)
                # MoGe-only baseline: run MoGe directly, no DKT
                moge_d = moge_depths(fr_c)
                moge_depth = np.where(moge_d > 1e-6, moge_d, args.max_depth).astype(np.float32)
                moge_depth = np.clip(moge_depth, 1e-3, args.max_depth)
                mf = [compute_metrics(moge_depth[t], gt_c[t], eval_mask_c[t])
                      for t in range(len(fr_c)) if eval_mask_c[t].sum() > 0]
                scene_per_frame_moge.extend(mf)
            else:
                pred_depth = transfer_pred_disp2depth(pred_disp, gt_c, eval_mask_c)

            pred_depth = np.clip(pred_depth, 1e-3, args.max_depth)

            if not clip_logged:
                clip_frac = float(np.mean((pred_depth <= 1e-3) | (pred_depth >= args.max_depth)))
                print(f"clip_frac={clip_frac:.4f} (max_depth={args.max_depth})")
                clip_logged = True

            per_frame = [
                compute_metrics(pred_depth[t], gt_c[t], eval_mask_c[t])
                for t in range(len(fr_c))
                if eval_mask_c[t].sum() > 0
            ]
            scene_per_frame.extend(per_frame)

        if not scene_per_frame:
            continue

        ms = np.mean(scene_per_frame, axis=0)
        all_m.append(ms)
        line = (f"[{si+1}/{len(scenes)}] {os.path.basename(scene)}  "
                f"REL={ms[0]:.2f} RMSE={ms[1]:.3f} "
                f"d1.05={ms[2]:.2f} d1.10={ms[3]:.2f} d1.25={ms[4]:.2f}")
        if args.metric_eval and scene_per_frame_moge:
            mm = np.mean(scene_per_frame_moge, axis=0)
            all_m_moge.append(mm)
            line += (f"  |  MoGe-only: REL={mm[0]:.2f} RMSE={mm[1]:.3f} "
                     f"d1.05={mm[2]:.2f}")
        print(line)

    if not all_m:
        print("no scenes evaluated -- check --clearpose_root / --depth_suffix with --dry_run")
        return
    A = np.mean(all_m, axis=0)
    print("\n==== OVERALL (mean over scenes) ====")
    print(f"REL         : {A[0]:.2f}\nRMSE        : {A[1]:.3f}")
    print(f"delta<1.05  : {A[2]:.2f}\ndelta<1.10  : {A[3]:.2f}\ndelta<1.25  : {A[4]:.2f}")
    if args.metric_eval and all_m_moge:
        B = np.mean(all_m_moge, axis=0)
        print("\n==== MoGe-only baseline (metric protocol) ====")
        print(f"REL         : {B[0]:.2f}\nRMSE        : {B[1]:.3f}")
        print(f"delta<1.05  : {B[2]:.2f}\ndelta<1.10  : {B[3]:.2f}\ndelta<1.25  : {B[4]:.2f}")


if __name__ == "__main__":
    main()

"""Diagnose ClearPose label semantics: which instance IDs are transparent objects.

Usage:
    # 1-frame visualization (default: scene1 frame 100)
    python tools/diag_label_viz.py

    # Multi-frame coverage check with a transparent-id whitelist
    python tools/diag_label_viz.py --transparent_class_ids 24,26,21,47 \\
            --num_frames 4 --frame_ids 0,50,100,150
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np


# Fixed bright palette (BGR for cv2) for ≤32 unique IDs
_PALETTE_BGR = np.array([
    (0, 0, 0),         (255, 0, 0),     (0, 255, 0),     (0, 0, 255),
    (255, 255, 0),     (255, 0, 255),   (0, 255, 255),   (128, 128, 0),
    (128, 0, 128),     (0, 128, 128),   (255, 128, 0),   (255, 0, 128),
    (128, 255, 0),     (0, 255, 128),   (128, 0, 255),   (0, 128, 255),
    (255, 64, 64),     (64, 255, 64),   (64, 64, 255),   (255, 255, 128),
    (255, 128, 255),   (128, 255, 255), (192, 64, 0),    (0, 192, 64),
    (64, 0, 192),      (192, 192, 0),   (192, 0, 192),   (0, 192, 192),
    (255, 192, 128),   (128, 255, 192), (192, 128, 255), (128, 192, 255),
], dtype=np.uint8)


def colorize_label(label: np.ndarray) -> np.ndarray:
    """label [H, W] uint → BGR [H, W] uint8 with deterministic per-ID color."""
    flat = label.ravel()
    colors = _PALETTE_BGR[np.clip(flat, 0, len(_PALETTE_BGR) - 1)]
    return colors.reshape(label.shape + (3,))


def per_class_stats(label: np.ndarray, gt_depth_mm: np.ndarray):
    ids = np.unique(label)
    rows = []
    for cid in ids:
        if cid == 0:
            continue
        m = label == cid
        if m.sum() < 50:
            continue
        gt_valid = gt_depth_mm[m] > 0
        med = float(np.median(gt_depth_mm[m][gt_valid])) if gt_valid.any() else 0.0
        rows.append((int(cid), int(m.sum()), float(m.mean()) * 100.0, med))
    return rows


def visualize_frame(color: np.ndarray, label: np.ndarray,
                    depth_mm: np.ndarray, transparent_ids: set) -> np.ndarray:
    """Build a 3-panel: color | label overlay | depth-colorized with mask."""
    h, w = color.shape[:2]

    # Panel 1: color
    p1 = color.copy()

    # Panel 2: label overlay (colored on top of color)
    label_bgr = colorize_label(label)
    overlay = cv2.addWeighted(color, 0.5, label_bgr, 0.5, 0)
    # Bold-red borders for transparent IDs
    if transparent_ids:
        for cid in transparent_ids:
            m = (label == cid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    p2 = overlay

    # Panel 3: depth colorized, with transparent mask highlighted
    d = depth_mm.astype(np.float32)
    d_valid = d > 0
    if d_valid.any():
        d_norm = np.zeros_like(d, dtype=np.float32)
        d_norm[d_valid] = (d[d_valid] - d[d_valid].min()) / (
            np.ptp(d[d_valid]) + 1e-6)
    else:
        d_norm = d
    cm = cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    p3 = cm.copy()
    if transparent_ids:
        m_obj = np.isin(label, list(transparent_ids))
        p3[m_obj] = (0, 0, 255)  # red on transparent region

    # Stack horizontally
    sep = np.full((h, 4, 3), 255, dtype=np.uint8)
    out = np.hstack([p1, sep, p2, sep, p3])
    return out


def coverage_check(scene_dir: str, frame_ids: list,
                   transparent_ids: set) -> dict:
    """For each frame, compute:
       - naive coverage (label > 0)
       - whitelisted coverage (label in transparent_ids)
       - union across requested frames (what dkt_temporal_moge_anchor would see)
    """
    naive_union = None
    white_union = None
    per_frame = []
    for fid in frame_ids:
        lbl_path = os.path.join(scene_dir, f"{fid:06d}-label.png")
        lbl = cv2.imread(lbl_path, cv2.IMREAD_UNCHANGED)
        if lbl is None:
            print(f"  WARN: {lbl_path} not found, skip")
            continue
        naive = lbl > 0
        white = np.isin(lbl, list(transparent_ids)) if transparent_ids else naive
        per_frame.append((fid, naive.mean() * 100, white.mean() * 100))
        if naive_union is None:
            naive_union = naive.copy()
            white_union = white.copy()
        else:
            naive_union |= naive
            white_union |= white
    return {
        "per_frame": per_frame,
        "naive_union_pct": float(naive_union.mean() * 100) if naive_union is not None else 0.0,
        "white_union_pct": float(white_union.mean() * 100) if white_union is not None else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="/workspace/datasets/clearpose/set2/scene1")
    ap.add_argument("--frame_id", type=int, default=100,
                    help="frame id for the visualization panel")
    ap.add_argument("--out", default="/workspace/DKT/debug_vis/scene1_frame100.png")
    ap.add_argument("--transparent_class_ids", default="",
                    help="comma-separated IDs to highlight as transparent; "
                         "empty = no whitelist")
    ap.add_argument("--num_frames", type=int, default=4)
    ap.add_argument("--frame_ids", default="0,50,100,150",
                    help="comma-separated frame ids for coverage check")
    args = ap.parse_args()

    transparent_ids = set()
    if args.transparent_class_ids:
        transparent_ids = {int(x) for x in args.transparent_class_ids.split(",")}

    scene = Path(args.scene)
    color = cv2.imread(str(scene / f"{args.frame_id:06d}-color.png"))
    label = cv2.imread(str(scene / f"{args.frame_id:06d}-label.png"),
                       cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(str(scene / f"{args.frame_id:06d}-depth_true.png"),
                       cv2.IMREAD_UNCHANGED)
    if color is None or label is None or depth is None:
        raise FileNotFoundError(f"missing files in {scene} for frame {args.frame_id}")

    # --- Visualization ---
    print(f"=== Per-class stats (frame {args.frame_id}) ===")
    print(f"  {'ID':>4} {'pixels':>8} {'frac%':>7} {'gt_med_mm':>10}")
    rows = per_class_stats(label, depth)
    for cid, cnt, frac, med in rows:
        marker = "  <-- transparent" if cid in transparent_ids else ""
        print(f"  {cid:>4} {cnt:>8} {frac:>6.2f}% {med:>10.1f}{marker}")
    print(f"  bg0: {(label==0).sum()} px ({(label==0).mean()*100:.1f}%)")
    if transparent_ids:
        union_obj = np.isin(label, list(transparent_ids))
        print(f"  transparent-union this frame: {union_obj.mean()*100:.2f}%")

    vis = visualize_frame(color, label, depth, transparent_ids)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"\nWrote visualization: {args.out}")

    # --- Coverage check ---
    fids = [int(x) for x in args.frame_ids.split(",")][: args.num_frames]
    print(f"\n=== Coverage check on frames {fids} ===")
    res = coverage_check(str(scene), fids, transparent_ids)
    print(f"  {'frame':>6} {'naive_%':>8} {'whitelist_%':>12}")
    for fid, n, w in res["per_frame"]:
        print(f"  {fid:>6} {n:>7.2f}% {w:>11.2f}%")
    print(f"\n  UNION across these frames (what dkt_temporal_moge_anchor sees):")
    print(f"    naive (label>0):        {res['naive_union_pct']:.2f}%")
    print(f"    whitelist (in IDs):    {res['white_union_pct']:.2f}%")
    if transparent_ids and res["white_union_pct"] > 50.0:
        print(f"  ⚠️  whitelist still > 50% — IDs likely include non-transparent classes")


if __name__ == "__main__":
    main()

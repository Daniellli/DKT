"""Object-centric Temporal Geometry Refinement for transparent object video depth.

Model per frame t:
    d_ref_t(x) = a_t * dkt_disp_t(x) + b_t + c_t * O_t(x)

    O_t(x) : 1 if x is a transparent-object pixel (label > 0), else 0
    a_t, b_t : affine params fitted on background pixels (label == 0) vs MoGe
    c_t      : robust-median residual of (a_t*pred+b_t) vs MoGe on object pixels

All three params are independently median-filtered over time, then applied.
MoGe is only used as a metric anchor for background; transparent regions are NOT
forced to match MoGe—only the temporally-smoothed c_t is added, which acts as
a noise-reduced correction rather than a hard alignment.

Fallback: if label_maps is None → delegates to moge_affine_mask.
"""

import numpy as np
import torch
from scipy.ndimage import median_filter, uniform_filter1d

from tools.refine_moge import (
    _infer_moge,
    refine_moge_affine_mask,
)


# ---------------------------------------------------------------------------
# Core fitting helpers
# ---------------------------------------------------------------------------

def fit_background_affine(pred_disp: np.ndarray,
                          moge_disp: np.ndarray,
                          bg_mask: np.ndarray,
                          trim_pct: float = 10.0):
    """
    Robust affine OLS on background pixels: pred*a + b ≈ moge_disp.

    Two-pass: initial OLS → residual trimming (drop outer trim_pct% on each
    side) → re-fit on inliers.  Returns (a, b).
    """
    p = pred_disp[bg_mask].ravel()
    r = moge_disp[bg_mask].ravel()
    if p.size < 50:
        return 1.0, 0.0

    # Pass 1
    A = np.stack([p, np.ones_like(p)], axis=1)
    res0, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
    a0, b0 = float(res0[0]), float(res0[1])

    # Residual trimming
    residuals = np.abs(r - (a0 * p + b0))
    lo = np.percentile(residuals, trim_pct)
    hi = np.percentile(residuals, 100.0 - trim_pct)
    inlier = (residuals >= lo) & (residuals <= hi)
    if inlier.sum() < 20:
        return a0, b0

    # Pass 2 on inliers
    A2 = np.stack([p[inlier], np.ones_like(p[inlier])], axis=1)
    res2, _, _, _ = np.linalg.lstsq(A2, r[inlier], rcond=None)
    return float(res2[0]), float(res2[1])


def estimate_object_correction(pred_disp: np.ndarray,
                               moge_disp: np.ndarray,
                               obj_mask: np.ndarray,
                               a: float, b: float,
                               trim_pct: float = 10.0) -> float:
    """
    Robust-median residual of (a*pred+b) vs moge_disp over object pixels.

    c > 0  →  MoGe predicts higher disparity (closer) than calibrated DKT.
    c < 0  →  calibrated DKT predicts closer than MoGe.

    Note: MoGe typically sees the background *through* transparent objects,
    so |c| is expected to be larger and noisier on transparent regions.
    Temporal smoothing of c reduces this per-frame noise.
    """
    if obj_mask.sum() < 10:
        return 0.0

    residual = moge_disp[obj_mask] - (a * pred_disp[obj_mask] + b)
    lo = np.percentile(residual, trim_pct)
    hi = np.percentile(residual, 100.0 - trim_pct)
    trimmed = residual[(residual >= lo) & (residual <= hi)]
    if trimmed.size == 0:
        return float(np.median(residual))
    return float(np.median(trimmed))


def temporal_smooth_params(a_seq: np.ndarray,
                           b_seq: np.ndarray,
                           c_seq: np.ndarray,
                           kernel: int = 5):
    """Independent median filter on each parameter sequence."""
    a_s = median_filter(a_seq, size=kernel, mode="nearest")
    b_s = median_filter(b_seq, size=kernel, mode="nearest")
    c_s = median_filter(c_seq, size=kernel, mode="nearest")
    return a_s, b_s, c_s


# ---------------------------------------------------------------------------
# Main refinement function
# ---------------------------------------------------------------------------

def refine_object_temporal(pred_disp: np.ndarray,
                           frames,
                           label_maps=None,
                           device: str = None,
                           temporal_window: int = 11) -> np.ndarray:
    """
    pred_disp      : [T, H, W]  DKT relative disparity
    frames         : list of T PIL Images
    label_maps     : [T, H, W] uint8 or None (ClearPose: 0=bg, >0=object id)
    temporal_window: median-filter kernel size over time
    returns        : [T, H, W] refined disparity, clipped to (1e-6, inf)
    """
    if label_maps is None:
        # No object masks available → fall back to standard MoGe affine refine
        return refine_moge_affine_mask(pred_disp, frames,
                                      label_maps=None, device=device)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    T = len(frames)
    a_seq = np.ones(T,  dtype=np.float64)
    b_seq = np.zeros(T, dtype=np.float64)
    c_seq = np.zeros(T, dtype=np.float64)

    for t, frame in enumerate(frames):
        moge_depth, moge_valid = _infer_moge(frame, device)

        moge_disp = np.zeros_like(moge_depth)
        pos = moge_valid & (moge_depth > 0)
        moge_disp[pos] = 1.0 / moge_depth[pos]

        raw_obj  = label_maps[t] > 0
        bg_mask  = (~raw_obj) & moge_valid & (pred_disp[t] > 0)
        obj_mask =   raw_obj  & moge_valid & (pred_disp[t] > 0)

        a_t, b_t = fit_background_affine(pred_disp[t], moge_disp, bg_mask)
        c_t = estimate_object_correction(pred_disp[t], moge_disp, obj_mask, a_t, b_t)

        a_seq[t], b_seq[t], c_seq[t] = a_t, b_t, c_t

    a_s, b_s, c_s = temporal_smooth_params(a_seq, b_seq, c_seq,
                                           kernel=temporal_window)

    print(f"[object_temporal] a: median={np.median(a_s):.4f} "
          f"min={a_s.min():.4f} max={a_s.max():.4f}")
    print(f"[object_temporal] b: median={np.median(b_s):.6f} "
          f"min={b_s.min():.6f} max={b_s.max():.6f}")
    print(f"[object_temporal] c: median={np.median(c_s):.6f} "
          f"min={c_s.min():.6f} max={c_s.max():.6f}")

    # Apply: background uses (a,b), object pixels additionally get c
    refined = np.empty_like(pred_disp, dtype=np.float64)
    for t in range(T):
        base = a_s[t] * pred_disp[t] + b_s[t]
        obj  = label_maps[t] > 0
        refined[t] = np.where(obj, base + c_s[t], base)

    return np.clip(refined, 1e-6, None).astype(pred_disp.dtype)


# ---------------------------------------------------------------------------
# A-road ablation: moge_affine_mask + temporal median on transparent regions
# ---------------------------------------------------------------------------

def moge_affine_mask_median(pred_disp: np.ndarray,
                            frames,
                            label_maps=None,
                            device: str = None,
                            median_kernel: int = 5) -> np.ndarray:
    """Ablation baseline (A-road).

    Re-uses `moge_affine_mask` exactly: per-frame (a_t, b_t) fit on background,
    temporally smoothed via uniform_filter1d, then ``refined = a*pred + b``.

    The only difference is that, on top of that, transparent-object pixels
    (label > 0) are additionally smoothed along the time axis with a
    median filter (default kernel=5).  Background regions are left exactly
    as the affine refinement produced them.

    Goal: show that "just add a median filter on the output" does not help —
    the structure-aware, MoGe-anchored smoothing in `dkt_temporal_moge_anchor`
    is what drives any improvement.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Per-frame (a_t, b_t) on background, identical to moge_affine_mask
    if label_maps is None:
        # Without masks, fall back to heuristic mask (transparent regions
        # are not known — smoothing would be a no-op anyway).
        return refine_moge_affine_mask(pred_disp, frames, label_maps=None,
                                       device=device)

    T = len(frames)
    raw_scales  = np.ones(T,  dtype=np.float64)
    raw_shifts  = np.zeros(T, dtype=np.float64)

    for t, frame in enumerate(frames):
        moge_depth, moge_valid = _infer_moge(frame, device)
        moge_disp = np.zeros_like(moge_depth)
        pos = moge_valid & (moge_depth > 0)
        moge_disp[pos] = 1.0 / moge_depth[pos]

        opaque = (label_maps[t] == 0) & moge_valid & (pred_disp[t] > 0)

        A = np.stack([pred_disp[t][opaque].ravel(),
                      np.ones(opaque.sum(), dtype=np.float64)], axis=1)
        r = moge_disp[opaque].ravel()
        if A.shape[0] >= 50:
            res, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
            raw_scales[t], raw_shifts[t] = float(res[0]), float(res[1])

    scales = uniform_filter1d(raw_scales, size=5, mode="nearest")
    shifts = uniform_filter1d(raw_shifts, size=5, mode="nearest")

    # 2) Apply affine (same as moge_affine_mask)
    refined = pred_disp * scales[:, None, None] + shifts[:, None, None]

    # 3) EXTRA: median-filter the refined disparity along time, but only on
    #          the union of transparent-object pixels.  Background is left
    #          exactly as the affine refinement produced it.
    obj_global = np.any(label_maps > 0, axis=0)  # [H, W] bool
    if obj_global.any():
        # Apply median_filter directly to the affine-refined stack.  At a
        # pixel that is transparent in some frames and background in
        # others, the median naturally mixes both — that is the desired
        # behaviour (it denoises both kinds of DKT output).  We only use
        # the smoothed value where obj_global is True.
        smoothed = median_filter(refined, size=(median_kernel, 1, 1),
                                 mode="nearest")
        refined = np.where(obj_global[None, :, :], smoothed, refined)

    return np.clip(refined, 1e-6, None).astype(pred_disp.dtype)


# ---------------------------------------------------------------------------
# B-road: dkt_temporal_moge_anchor
# ---------------------------------------------------------------------------

def dkt_temporal_moge_anchor(pred_disp: np.ndarray,
                             frames,
                             label_maps=None,
                             device: str = None,
                             median_kernel: int = 5) -> np.ndarray:
    """DKT-structure + MoGe-anchor temporal refinement (B-road).

    Pipeline:
      1. For each frame t, fit a_t, b_t on background pixels:
             a_t * dkt_disp[t] + b_t  ≈  moge_disp[t]
         (MoGe is only consulted for the metric anchor on background; it
         is NOT used as a fitting target on transparent regions.)
      2. Stack the raw DKT disparities into [T, H, W].  Compute the union
         of transparent-object masks across all frames: obj_global.
      3. Median-filter the stack along the time axis with kernel
         (median_kernel, 1, 1), but ONLY on obj_global.  Background pixels
         are left as the raw DKT output (no temporal mixing).
      4. Compose: refined[t] = a_t * dkt_disp_smooth[t] + b_t, then
         disparity → depth via the standard 1/disp inversion downstream.

    Key invariant: the structure of transparent regions comes from DKT
    (temporally smoothed), NOT from MoGe.  MoGe contributes only the
    metric scale/shift a_t, b_t fit on background.
    """
    if label_maps is None:
        # No object masks → behaviour is degenerate (everything is "bg"),
        # fall back to the proven MoGe affine refinement.
        return refine_moge_affine_mask(pred_disp, frames, label_maps=None,
                                       device=device)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    T = pred_disp.shape[0]
    a_seq = np.ones(T,  dtype=np.float64)
    b_seq = np.zeros(T, dtype=np.float64)

    for t, frame in enumerate(frames):
        moge_depth, moge_valid = _infer_moge(frame, device)
        moge_disp = np.zeros_like(moge_depth)
        pos = moge_valid & (moge_depth > 0)
        moge_disp[pos] = 1.0 / moge_depth[pos]

        bg_mask  = (label_maps[t] == 0) & moge_valid & (pred_disp[t] > 0)
        a_t, b_t = fit_background_affine(pred_disp[t], moge_disp, bg_mask)
        a_seq[t], b_seq[t] = a_t, b_t

    # 1) Smooth DKT disparity along time, transparent regions only.
    obj_global = np.any(label_maps > 0, axis=0)  # [H, W] bool
    dkt_smooth = pred_disp.astype(np.float64, copy=True)
    if obj_global.any() and T >= 2:
        # Apply median_filter directly to the raw dkt_disp stack.  This
        # preserves DKT's structure at every pixel (whether the pixel is
        # transparent or background in a given frame); the median naturally
        # smooths frame-to-frame noise.  Background pixels are restored
        # from the original dkt_disp below (we only use the smoothed value
        # on the union of transparent pixels).
        smoothed = median_filter(dkt_smooth,
                                 size=(median_kernel, 1, 1),
                                 mode="nearest")
        dkt_smooth = np.where(obj_global[None, :, :], smoothed, pred_disp)

    # 2) Compose: refined[t] = a_t * dkt_smooth[t] + b_t
    refined = a_seq[:, None, None] * dkt_smooth + b_seq[:, None, None]

    print(f"[dkt_temporal_moge_anchor] a: median={np.median(a_seq):.4f} "
          f"min={a_seq.min():.4f} max={a_seq.max():.4f}")
    print(f"[dkt_temporal_moge_anchor] b: median={np.median(b_seq):.6f} "
          f"min={b_seq.min():.6f} max={b_seq.max():.6f}")
    print(f"[dkt_temporal_moge_anchor] obj_global coverage: "
          f"{obj_global.mean()*100:.2f}% of pixels")

    return np.clip(refined, 1e-6, None).astype(pred_disp.dtype)

"""MoGe-guided per-frame scale alignment for DKT.

Public API (called by eval_clearpose.py via --refine):
  refine_none()            → identity, DKT baseline
  refine_moge_scale()      → per-frame scale-only OLS (no shift)
  refine_moge_affine()     → per-frame affine OLS, heuristic opaque mask
  refine_moge_affine_mask()→ per-frame affine OLS, ClearPose label mask  ← current best

All share the same signature:
  fn(pred_disp, frames, label_maps=None, device=None) → refined_disp
"""

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

_moge_cache = None


def _get_moge(device: str):
    global _moge_cache
    if _moge_cache is None:
        from moge.model.v2 import MoGeModel
        cached = "checkpoints/moge_ckpt/moge-2-vitl-normal/model.pt"
        _moge_cache = MoGeModel.from_pretrained(cached).to(device).eval()
    return _moge_cache


def _infer_moge(frame, device: str):
    """PIL Image → (depth_m [H,W] float32, valid_mask [H,W] bool)."""
    model = _get_moge(device)
    img_np = np.array(frame).astype(np.float32) / 255.0
    t = torch.from_numpy(img_np).permute(2, 0, 1).to(device)
    with torch.no_grad():
        out = model.infer(t)
    depth = out["depth"].cpu().numpy().astype(np.float32)
    mask = out["mask"].cpu().numpy().astype(bool)
    return depth, mask


def _robust_mask(pred_disp, moge_disp, moge_valid, keep_pct=75):
    """
    Pixels where DKT and MoGe agree in normalised structure.
    Disagreeing pixels are likely transparent → MoGe unreliable there.
    """
    valid = moge_valid & (pred_disp > 0)
    if not valid.any():
        return valid

    def _norm(x):
        v = x[valid]
        mn, mx = v.min(), v.max()
        out = np.zeros_like(x)
        if mx > mn:
            out[valid] = (x[valid] - mn) / (mx - mn)
        return out

    diff = np.abs(_norm(pred_disp) - _norm(moge_disp))
    thresh = np.percentile(diff[valid], keep_pct)
    return valid & (diff <= thresh)


def _scale_ols(pred_disp, ref_disp, mask):
    """Scale-only OLS: argmin_s ||pred*s - ref||²  →  s = <pred,ref>/<pred,pred>."""
    p = pred_disp[mask].ravel()
    r = ref_disp[mask].ravel()
    if p.size < 50:
        return 1.0
    return float(np.dot(p, r) / (np.dot(p, p) + 1e-12))


def _affine_ols(pred_disp, ref_disp, mask):
    """Affine OLS in disparity space: argmin_{s,b} ||pred*s + b - ref||²
    Returns (scale, shift). pred_disp has a non-zero shift offset so
    scale-only is insufficient; affine properly maps DKT → MoGe units."""
    p = pred_disp[mask].ravel()
    r = ref_disp[mask].ravel()
    if p.size < 50:
        return 1.0, 0.0
    A = np.stack([p, np.ones_like(p)], axis=1)
    result, _, _, _ = np.linalg.lstsq(A, r, rcond=None)
    return float(result[0]), float(result[1])


def moge_depths(frames, device: str = None) -> np.ndarray:
    """Run MoGe only on each frame → [T, H, W] metric depth in meters (0 = invalid)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = []
    for frame in frames:
        d, m = _infer_moge(frame, device)
        d[~m] = 0.0
        result.append(d.astype(np.float32))
    return np.stack(result)


# ---------------------------------------------------------------------------
# Public named-refine API  (all share the same call signature)
# ---------------------------------------------------------------------------

def refine_none(pred_disp: np.ndarray, frames,
                label_maps=None, device: str = None) -> np.ndarray:
    """Identity: returns DKT disparity unchanged (pure baseline)."""
    return pred_disp


def refine_moge_scale(pred_disp: np.ndarray, frames,
                      label_maps=None, device: str = None) -> np.ndarray:
    """Scale-only OLS per frame (no shift). Opaque mask: label or heuristic."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    T = len(frames)
    raw_scales = np.ones(T, dtype=np.float64)
    for t, frame in enumerate(frames):
        moge_depth, moge_valid = _infer_moge(frame, device)
        moge_disp = np.zeros_like(moge_depth)
        pos = moge_valid & (moge_depth > 0)
        moge_disp[pos] = 1.0 / moge_depth[pos]
        if label_maps is not None:
            opaque = (label_maps[t] == 0) & moge_valid & (pred_disp[t] > 0)
        else:
            opaque = _robust_mask(pred_disp[t], moge_disp, moge_valid)
        raw_scales[t] = _scale_ols(pred_disp[t], moge_disp, opaque)
    scales = uniform_filter1d(raw_scales, size=5, mode="nearest")
    refined = pred_disp * scales[:, None, None]
    return np.clip(refined, 1e-6, None).astype(pred_disp.dtype)


def refine_moge_affine(pred_disp: np.ndarray, frames,
                       label_maps=None, device: str = None) -> np.ndarray:
    """Affine OLS per frame, heuristic opaque mask (ignores label_maps)."""
    return refine(pred_disp, frames, label_maps=None, device=device)


def refine_moge_affine_mask(pred_disp: np.ndarray, frames,
                            label_maps=None, device: str = None) -> np.ndarray:
    """Affine OLS per frame, ClearPose label mask for opaque region. Current best."""
    return refine(pred_disp, frames, label_maps=label_maps, device=device)


# ---------------------------------------------------------------------------
# Internal implementation (used by the public functions above)
# ---------------------------------------------------------------------------

def refine(pred_disp: np.ndarray, frames,
           label_maps=None, device: str = None) -> np.ndarray:
    """
    pred_disp  : [T, H, W]  DKT relative disparity in [0, 1]
    frames     : list of T  PIL Images
    label_maps : [T, H, W] uint8 or None. ClearPose label: 0=background,
                 >0=transparent object. When provided, only background pixels
                 are used for scale estimation (Priority-3 exact opaque mask).
                 Falls back to heuristic robust_mask when None.
    returns    : [T, H, W]  scale-corrected disparity (same dtype)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    T = len(frames)
    raw_scales = np.ones(T, dtype=np.float64)
    raw_shifts = np.zeros(T, dtype=np.float64)

    for t, frame in enumerate(frames):
        moge_depth, moge_valid = _infer_moge(frame, device)

        moge_disp = np.zeros_like(moge_depth)
        pos = moge_valid & (moge_depth > 0)
        moge_disp[pos] = 1.0 / moge_depth[pos]

        if label_maps is not None:
            # Priority-3: exact opaque mask from ClearPose labels
            opaque = (label_maps[t] == 0) & moge_valid & (pred_disp[t] > 0)
        else:
            opaque = _robust_mask(pred_disp[t], moge_disp, moge_valid)

        s, b = _affine_ols(pred_disp[t], moge_disp, opaque)
        raw_scales[t], raw_shifts[t] = s, b

    # Priority-2: temporal smoothing on both scale and shift sequences
    scales = uniform_filter1d(raw_scales, size=5, mode="nearest")
    shifts = uniform_filter1d(raw_shifts, size=5, mode="nearest")

    refined = pred_disp * scales[:, None, None] + shifts[:, None, None]
    refined = np.clip(refined, 1e-6, None)
    return refined.astype(pred_disp.dtype)

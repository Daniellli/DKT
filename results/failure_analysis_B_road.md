# B-road Failure Analysis: Why DKT-temporal-MoGe-anchor underperforms

## Design

`dkt_temporal_moge_anchor` was designed as a refinement on `moge_affine_mask` by adding
per-pixel temporal median smoothing of DKT disparity, restricted to per-frame transparent
mask (`label > 0`), followed by the same per-frame background-anchored affine alignment
as `moge_affine_mask`. The intuition: DKT provides relative structure, MoGe provides
metric scale, temporal smoothing reduces transparent-region noise.

## Result

5-scene average d1.05: 41.77 (vs `moge_affine_mask` 44.92, **−3.15**).
5-scene average RMSE: 0.153 m (vs `moge_affine_mask` 0.150 m, **+3 mm**).
**B-road loses to A-road on all 5 scenes**: 50.40 < 54.72, 46.69 < 50.31, 38.62 < 42.97, 42.12 < 43.59, 31.05 < 33.00.

## Diagnosis

Two compounding causes:

1. **Per-frame mask coverage is too broad.** On ClearPose set2, `label > 0` covers
   30–50% of pixels per frame (includes hands, gripper, table-edge objects, not only
   transparent items). Median over time within this region blurs DKT's correct relative
   structure on non-transparent foreground.
2. **Temporal smoothing operates on disparity, while the metric anchor is depth.**
   Median in disparity space, followed by `disp → depth` inversion, amplifies error
   in low-disparity (far) regions and produces no measurable RMSE gain.

## Implication for A-road

The `moge_affine_mask_median` ablation (+0.12 d1.05 vs `moge_affine_mask`) confirms that
median temporal smoothing has no effect when applied to the already-lstsq-aligned depth
on the same dense mask. **Per-frame least-squares is already optimal at this protocol;
temporal regularization adds nothing.**

## Direction not pursued

A potentially salvageable variant — **optical-flow-warped temporal alignment of
transparent pixels** — was deferred. Implementation cost is ~1 week and expected gain
is bounded by the same lstsq saturation observed in A-road ablation. Resources allocated
to ② (full transparent 3D reconstruction) instead.

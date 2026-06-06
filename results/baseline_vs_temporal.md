# ① Metric Depth Refinement on ClearPose Set2 — Baseline Comparison

**Protocol** (replicated historical): `--max_frames 200 --depth_scale 0.001 --max_depth 2.0`,
5 scenes (scene 1/3/4/5/6), DKT-1.3B + MoGe-2-vitL, training-free.

## Main result (5-scene average)

| Method                       | d1.05  | d1.10  | d1.25  | REL%   | RMSE(m) | Notes                             |
|------------------------------|--------|--------|--------|--------|---------|-----------------------------------|
| none (DKT raw)               | 40.93  | 67.74  | 94.01  | 8.81   | 0.154   | baseline                          |
| moge_affine                  | —      | —      | —      | —      | —       | dropped (mask is strict superset) |
| **moge_affine_mask**         | **44.92** | **72.44** | **95.44** | **8.19** | **0.150** | **main method** (+3.99 d1.05, −4 mm RMSE) |
| moge_affine_mask_median      | 45.04  | 72.48  | 95.42  | 8.19   | 0.150   | A-road ablation (+0.12 vs mask, within noise → median has no effect on dense per-frame mask) |
| dkt_temporal_moge_anchor     | 41.77  | 69.40  | 95.18  | 8.63   | 0.153   | B-road, failed (see failure_analysis_B_road.md) |

## Per-scene (d1.05)

| Method                       | scene1 | scene3 | scene4 | scene5 | scene6 |
|------------------------------|--------|--------|--------|--------|--------|
| none                         | 45.66  | 50.40  | 49.75  | 27.09  | 31.77  |
| moge_affine_mask             | 54.72  | 50.31  | 42.97  | 43.59  | 33.00  |
| moge_affine_mask_median      | 55.06  | 50.32  | 43.16  | 43.62  | 33.02  |
| dkt_temporal_moge_anchor     | 50.40  | 46.69  | 38.62  | 42.12  | 31.05  |

## Per-scene (RMSE, mm)

| Method                       | scene1 | scene3 | scene4 | scene5 | scene6 |
|------------------------------|--------|--------|--------|--------|--------|
| none                         | 133    | 148    | 166    | 153    | 170    |
| moge_affine_mask             | 134    | 147    | 168    | 141    | 158    |
| moge_affine_mask_median      | 135    | 148    | 168    | 141    | 158    |
| dkt_temporal_moge_anchor     | 135    | 152    | 171    | 142    | 163    |

## Statement of claim (paper-ready)

We present a training-free pipeline that produces metric depth from monocular
transparent-object video by anchoring DKT's affine-invariant disparity to MoGe's
metric depth via per-frame least-squares regression on opaque (label==0) regions.
The method achieves **+3.99 d1.05** and **−4 mm RMSE** over the DKT baseline on
ClearPose set2 (5 scenes), without any training.

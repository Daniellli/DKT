"""
Standalone diagnostic: load ClearPose scene metadata for a given frame,
print K, T_world_cam, per-object (class_id, T_cam_obj), and check for normal GT.

Usage:
    python tools/recon3d/diag_meta.py <scene_dir> <frame_idx>
    python tools/recon3d/diag_meta.py /workspace/datasets/clearpose/set2/scene1 0
"""

import sys
import os
import numpy as np
import scipy.io as sio


def _unwrap(x):
    while hasattr(x, "shape") and x.ndim > 0 and x.size == 1:
        x = x.flat[0] if x.ndim == 1 else x[0, 0]
    return x


def load_frame(scene_dir: str, frame_idx: int):
    mat_path = os.path.join(scene_dir, "metadata.mat")
    mat = sio.loadmat(mat_path)
    key = f"{frame_idx:06d}"
    if key not in mat:
        raise KeyError(f"Frame {key} not in metadata.mat. Keys: {[k for k in mat if not k.startswith('_')][:5]}...")

    f = mat[key][0, 0]

    K = _unwrap(f["intrinsic_matrix"])           # (3,3)
    RT_cam = _unwrap(f["rotation_translation_matrix"])  # (3,4) camera extrinsic [R|t]
    poses = _unwrap(f["poses"])                  # (3,4,N) T_cam_obj per object
    cls_ids = _unwrap(f["cls_indexes"]).flatten()  # (N,)
    factor_depth = int(_unwrap(f["factor_depth"]).flat[0])

    # Camera world position: p_world = -R^T @ t
    R = RT_cam[:3, :3]
    t = RT_cam[:3, 3]
    cam_pos_world = -R.T @ t

    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R.T
    T_world_cam[:3, 3] = cam_pos_world

    objects = []
    n_obj = poses.shape[2] if poses.ndim == 3 else 1
    for i in range(n_obj):
        T_co = np.eye(4)
        T_co[:3, :] = poses[:, :, i] if poses.ndim == 3 else poses
        objects.append({"class_id": int(cls_ids[i]), "T_cam_obj": T_co})

    # Check normal GT
    normal_path = os.path.join(scene_dir, f"{frame_idx:06d}-normal_true.png")
    has_normal_gt = os.path.isfile(normal_path)

    return {
        "K": K,
        "RT_cam_extrinsic": RT_cam,
        "T_world_cam": T_world_cam,
        "cam_pos_world": cam_pos_world,
        "objects": objects,
        "factor_depth": factor_depth,
        "has_normal_gt": has_normal_gt,
        "normal_gt_path": normal_path if has_normal_gt else None,
    }


def main():
    if len(sys.argv) < 3:
        print("Usage: python tools/recon3d/diag_meta.py <scene_dir> <frame_idx>")
        sys.exit(1)

    scene_dir = sys.argv[1]
    frame_idx = int(sys.argv[2])

    data = load_frame(scene_dir, frame_idx)

    np.set_printoptions(precision=4, suppress=True)

    print(f"\n=== Frame {frame_idx:06d} @ {scene_dir} ===\n")

    print("K =")
    print(data["K"])

    print("\nT_world_cam (4x4, camera pose in world) =")
    print(data["T_world_cam"])

    print(f"\ncam_pos_world (xyz) = {data['cam_pos_world']}")

    print(f"\nfactor_depth = {data['factor_depth']}  (depth_m = raw_depth / factor_depth)")

    print(f"\nhas_normal_GT = {data['has_normal_gt']}")
    if data["has_normal_gt"]:
        print(f"  path: {data['normal_gt_path']}")

    print(f"\n{len(data['objects'])} objects:")
    for obj in data["objects"]:
        T = obj["T_cam_obj"]
        print(f"  class_id={obj['class_id']}  T_cam_obj=")
        print(f"    {T[0]}")
        print(f"    {T[1]}")
        print(f"    {T[2]}")
        print(f"    {T[3]}")


if __name__ == "__main__":
    main()

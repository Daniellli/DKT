"""ClearPose dataset helpers: scene discovery and sequence loading."""

import glob
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

DEFAULT_DEPTH_SUFFIXES = ("-depth_true.png", "-depth.png")


@dataclass
class FrameRecord:
    color_path: str
    depth_path: str
    mask_path: Optional[str] = None
    label_path: Optional[str] = None  # ClearPose label: 0=background, >0=transparent object


def discover_sequence_dirs(root: str, depth_suffixes: Tuple[str, ...] = DEFAULT_DEPTH_SUFFIXES) -> List[str]:
    scenes = []
    for d, _, files in os.walk(root):
        has_color = any(f.endswith("-color.png") for f in files)
        has_depth = any(f.endswith(s) for f in files for s in depth_suffixes)
        if has_color and has_depth:
            scenes.append(d)
    return sorted(scenes)


def discover_sequence_frames(
    scene_dir: str,
    depth_suffixes: Tuple[str, ...] = DEFAULT_DEPTH_SUFFIXES,
) -> List[FrameRecord]:
    color_paths = sorted(glob.glob(os.path.join(scene_dir, "*-color.png")))
    records = []
    for c in color_paths:
        stem = c[: -len("-color.png")]
        depth_path = None
        for s in depth_suffixes:
            candidate = stem + s
            if os.path.exists(candidate):
                depth_path = candidate
                break
        if depth_path is None:
            continue
        mask_candidate = stem + "-mask.png"
        mask_path = mask_candidate if os.path.exists(mask_candidate) else None
        label_candidate = stem + "-label.png"
        label_path = label_candidate if os.path.exists(label_candidate) else None
        records.append(FrameRecord(color_path=c, depth_path=depth_path,
                                   mask_path=mask_path, label_path=label_path))
    return records


def load_clearpose_sequence(
    scene_dir: str,
    depth_scale: float = 1.0,
    max_frames: int = 0,
    stride: int = 1,
    depth_suffix: Optional[str] = None,
) -> Tuple[List[Image.Image], np.ndarray, np.ndarray, List[FrameRecord]]:
    suffixes = DEFAULT_DEPTH_SUFFIXES
    if depth_suffix:
        suffixes = (depth_suffix,) + tuple(s for s in DEFAULT_DEPTH_SUFFIXES if s != depth_suffix)

    records = discover_sequence_frames(scene_dir, depth_suffixes=suffixes)
    records = records[::stride]
    if max_frames:
        records = records[:max_frames]

    if not records:
        return [], np.array([]), np.array([]), []

    frames, gts, masks = [], [], []
    used_records = []
    for rec in records:
        img = Image.open(rec.color_path).convert("RGB")
        gt = cv2.imread(rec.depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if gt is None:
            continue
        gt = gt * depth_scale
        frames.append(img)
        gts.append(gt)
        masks.append(gt > 0)
        used_records.append(rec)

    if not frames:
        return [], np.array([]), np.array([]), []

    return frames, np.stack(gts), np.stack(masks), used_records

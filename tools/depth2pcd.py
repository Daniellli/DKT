

import numpy as np
import open3d as o3d

def depth2pcd(depth, intrinsic, color=None, input_mask=None, ret_pcd=False):
    """
    Convert a depth map into a 3D point cloud.

    Args:
        depth (np.ndarray): (H, W) depth map in meters.
        intrinsic (np.ndarray): (3, 3) camera intrinsic matrix.
        color (np.ndarray, optional): (H, W, 3) RGB image aligned with the depth map.
        input_mask (np.ndarray, optional): (H, W) boolean mask indicating valid pixels.
        ret_pcd (bool, optional): If True, returns an Open3D PointCloud object;
                                  otherwise returns NumPy arrays.

    Returns:
        - If ret_pcd=True: returns `o3d.geometry.PointCloud()`
        - Otherwise: returns (N, 3) point coordinates and (N, 3) color arrays.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[0,2], intrinsic[1,2]
    
    Z = depth.reshape(-1)
    X = ((u.reshape(-1) - cx) / fx) * Z
    Y = ((v.reshape(-1) - cy) / fy) * Z

    points = np.stack([X, Y, Z], axis=1)

    # mask valid points
    mask = np.ones_like(Z, dtype=bool)
    if input_mask is not None:
        mask &= input_mask.reshape(-1)

    # Keep only valid points
    points = points[mask]

    # Process color information
    if color is not None:
        color = color.astype(np.float32) / 255.0
        colors = color.reshape(-1, 3)[mask]
    else:
        colors = None
    
    # Return Open3D point cloud or NumPy arrays
    if ret_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    else:
        return points, colors
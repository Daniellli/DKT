import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import io





def colorize_depth_map(depth, mask=None, reverse_color=False, color_map="Spectral"):
    cm = matplotlib.colormaps[color_map]
    #* the depth is normalized by defailt
    
    if reverse_color:
        img_colored_np = cm(1 - depth, bytes=False)[:, :, 0:3]
    else:
        img_colored_np = cm(depth, bytes=False)[:, :, 0:3]

    depth_colored = (img_colored_np * 255).astype(np.uint8)
    if mask is not None:
        masked_image = np.zeros_like(depth_colored)
        masked_image[mask] = depth_colored[mask]
        depth_colored_img = Image.fromarray(masked_image)
    else:
        depth_colored_img = Image.fromarray(depth_colored)
        
    return depth_colored_img




def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)




def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
            gt = downscaler(torch.as_tensor(gt).unsqueeze(0)).numpy()
            pred = downscaler(torch.as_tensor(pred).unsqueeze(0)).numpy()
            valid_mask = (
                downscaler(torch.as_tensor(valid_mask).unsqueeze(0).float())
                .bool()
                .numpy()
            )

    assert (
        gt.shape == pred.shape == valid_mask.shape
    ), f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred




def transfer_pred_disp2depth(all_pred_disparity, all_gt_depths, all_masks):
    gt_disparity,gt_non_neg_mask = depth2disparity(all_gt_depths, return_mask=True)
    pred_non_neg_mask = all_pred_disparity > 0
    valid_non_neg_mask = pred_non_neg_mask & gt_non_neg_mask & all_masks

    align_disp_pred,scale,shift = align_depth_least_square( gt_arr=gt_disparity,
            pred_arr=all_pred_disparity,
            valid_mask_arr=valid_non_neg_mask,
            return_scale_shift=True,
            max_resolution=None,)

    align_disp_pred = np.clip(
            align_disp_pred, a_min=1e-3, a_max=None
        )  # avoid 0 disparity
    all_pred_depths = disparity2depth(align_disp_pred)
    return all_pred_depths

    
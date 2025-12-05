"""
    reference:
    https://github.com/TencentARC/GeometryCrafter/blob/main/visualize/vis_point_maps.py

"""
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
from decord import cpu, VideoReader

from kornia.filters import canny
from kornia.morphology import dilation
import viser
import viser.extras
import viser.transforms as tf

def compute_edge(depth: torch.Tensor):
    magnitude, edges = canny(depth[None, None, :, :], low_threshold=0.4, high_threshold=0.5)
    magnitude = magnitude[0, 0]
    edges = edges[0, 0]
    return edges > 0

def dilation_mask(mask: torch.Tensor, edge_dilation_radius: int):
    mask = mask.float()
    mask = dilation(mask[None, None, :, :], torch.ones((edge_dilation_radius,edge_dilation_radius), device=mask.device))
    return mask[0, 0] > 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', required=True, type=str)
    parser.add_argument('--data_path', required=True, type=str)
    parser.add_argument("--indices", type=int, default=None, nargs='+', help='only load these frames for visualization')
    parser.add_argument("--sample_num", type=int, default=None, help='only sample several frames for visualization')
    parser.add_argument("--downsample_ratio", type=int, default=1, help='downsample ratio')
    parser.add_argument("--point_size", type=float, default=0.002, help='point size')  
    parser.add_argument("--scale_factor", type=float, default=1.0, help='point cloud scale factor for visualization')
    parser.add_argument("--edge_dilation_radius", type=int, default=3, help='remove floater points for visualization')
    parser.add_argument("--port", type=int, default=7891, help='port')
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True)
    point_map, mask = data['point_map'], data['mask']

    if args.indices:
        indices = np.array(args.indices, dtype=np.int32)
    elif args.sample_num:
        indices = np.linspace(0, len(point_map)-1, args.sample_num)
        indices = np.round(indices).astype(np.int32)
    else:
        indices = np.array(list(range(len(point_map))))
    
    point_map = torch.tensor(point_map[indices]).float()
    mask = torch.tensor(mask[indices]).bool()
    
    vid = VideoReader(args.video_path, ctx=cpu(0))
    frames = torch.tensor(vid.get_batch(indices).asnumpy()).float()

    H, W = point_map.shape[1:3]

    if frames.shape[1:3] != (H, W):
        frames = F.interpolate(frames.permute(0,3,1,2), (H, W), mode='bicubic').clamp(0, 255).permute(0,2,3,1)
        

    if args.downsample_ratio > 1:
        H, W = H // args.downsample_ratio, W // args.downsample_ratio
        point_map = F.interpolate(point_map.permute(0,3,1,2), (H, W)).permute(0,2,3,1)
        frames = F.interpolate(frames.permute(0,3,1,2), (H, W)).permute(0,2,3,1)
        mask = F.interpolate(mask.float()[:, None], (H, W))[:, 0] > 0.5

    if args.edge_dilation_radius > 0:
        for i in range(len(mask)):
            edge_mask = compute_edge(point_map[i, :, :, 2])
            edge_mask = dilation_mask(edge_mask, args.edge_dilation_radius)
            mask[i] = mask[i] & (~edge_mask)
    

    server = viser.ViserServer(port=args.port)
    server.request_share_url()
    num_frames = len(frames)

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=False,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=False)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=False)
        gui_playing = server.gui.add_checkbox("Playing", False)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=1, initial_value=round(vid.get_avg_fps())
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

     # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        global prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():

            # Toggle visibility.
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        # pcd, color = pcds[i], colors[i]

        position = point_map[i].reshape(-1, 3)[mask[i].reshape(-1)].numpy()
        color = frames[i].reshape(-1, 3)[mask[i].reshape(-1)].numpy().astype(np.uint8)

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(
            f"/frames/t{i}", 
            show_axes=False, 
            wxyz=tf.SO3.exp(np.array([0.0, 0.0, np.pi])).wxyz
        ))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position * args.scale_factor,
            colors=color,
            point_size=args.point_size * args.scale_factor,
            point_shape="rounded",
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)
        
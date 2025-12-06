import argparse
import numpy as np
from tqdm import tqdm
import time
from loguru import logger
import viser
import viser.transforms as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default='logs/tmp_save_np_small.npz', help='path to npz file')
    parser.add_argument("--indices", type=int, default=None, nargs='+', help='only load these frames for visualization')
    parser.add_argument("--sample_num", type=int, default=None, help='only sample several frames for visualization')
    parser.add_argument("--point_size", type=float, default=0.002, help='point size')  
    parser.add_argument("--scale_factor", type=float, default=1.0, help='point cloud scale factor for visualization')
    parser.add_argument("--port", type=int, default=7891, help='port')
    args = parser.parse_args()

    # Load data from npz file
    data = np.load(args.data_path, allow_pickle=True)
    point_cloud_data = data['arr_0']  # shape: [num_frame, num_pc, 6]

    if len(point_cloud_data.shape) == 4 and point_cloud_data.shape[1] == 1:
        point_cloud_data = np.squeeze(point_cloud_data, axis=1)
    logger.info(f"point_cloud_data.shape: {point_cloud_data.shape}")

    # Extract positions (first 3 dims) and colors (last 3 dims)
    positions = point_cloud_data[:, :, :3]  # [num_frame, num_pc, 3]
    colors = point_cloud_data[:, :, 3:6]   # [num_frame, num_pc, 3]
    
    num_frames = len(positions)

    if args.indices:
        indices = np.array(args.indices, dtype=np.int32)
    elif args.sample_num:
        indices = np.linspace(0, num_frames-1, args.sample_num)
        indices = np.round(indices).astype(np.int32)
    else:
        indices = np.array(list(range(num_frames)))
    
    positions = positions[indices]  # [selected_frames, num_pc, 3]
    colors = colors[indices]        # [selected_frames, num_pc, 3]
    
    # Normalize colors to [0, 255] if needed (assuming they might be in [0, 1] or [0, 255])
    if colors.max() <= 1.0:
        colors = (colors * 255).astype(np.uint8)
    else:
        colors = colors.astype(np.uint8)

    server = viser.ViserServer(port=args.port)
    server.request_share_url()
    num_frames = len(positions)

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
            "FPS", min=1, max=60, step=1, initial_value=30
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
        # Extract position and color for this frame
        position = positions[i]  # [num_pc, 3]
        color = colors[i]        # [num_pc, 3]

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
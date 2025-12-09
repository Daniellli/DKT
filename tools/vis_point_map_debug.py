import argparse
import numpy as np
from tqdm import tqdm
import time
from loguru import logger
import viser
import viser.transforms as tf
import imageio
from PIL import Image
import os
import threading

    
# Calculate initial camera position and orientation to look at point cloud center
def compute_camera_pose(position, look_at, up_direction=np.array([0, 0, 1])):
    """
    Compute camera quaternion (wxyz) to look at a target point.
    
    Args:
        position: Camera position (3D array)
        look_at: Target point to look at (3D array)
        up_direction: Up direction vector (default: +z)
    
    Returns:
        wxyz: Quaternion as tuple (w, x, y, z)
    """
    # Direction from camera to target (this is the direction camera should look)
    forward = look_at - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        # If camera is at the target, use default orientation
        return tf.SO3.identity().wxyz
    
    forward = forward / forward_norm
    
    # For OpenCV convention [+Z forward, +X right, +Y down]
    # We need: right = cross(up, forward), down = cross(forward, right)
    # Right vector: cross product of up and forward
    right = np.cross(up_direction, forward)
    right_norm = np.linalg.norm(right)
    
    # Handle case where forward and up are parallel
    if right_norm < 1e-6:
        # Use a different up direction (e.g., [1, 0, 0] or [0, 1, 0])
        alternative_up = np.array([1, 0, 0]) if abs(forward[0]) < 0.9 else np.array([0, 1, 0])
        right = np.cross(alternative_up, forward)
        right_norm = np.linalg.norm(right)
        if right_norm > 1e-6:
            right = right / right_norm
            down = np.cross(forward, right)
        else:
            # Fallback to identity
            return tf.SO3.identity().wxyz
    else:
        right = right / right_norm
        # Down vector: cross product of forward and right (for OpenCV convention)
        down = np.cross(forward, right)
    
    # Build rotation matrix for OpenCV convention [+Z forward, +X right, +Y down]
    # Rotation matrix columns: [right, down, forward]
    rotation_matrix = np.array([
        [right[0], down[0], forward[0]],  # X axis (right)
        [right[1], down[1], forward[1]],  # Y axis (down)
        [right[2], down[2], forward[2]]   # Z axis (forward)
    ])
    
    # Convert rotation matrix to quaternion (wxyz format)
    # Using viser's SO3 utilities
    so3 = tf.SO3.from_matrix(rotation_matrix)
    wxyz = so3.wxyz
    return wxyz

def downsample_point_cloud_random(points, colors, max_points):
    """
    Randomly downsample point cloud to max_points.
    
    Args:
        points: (N, 3) array of point positions
        colors: (N, 3) array of point colors
        max_points: Maximum number of points to keep
    
    Returns:
        downsampled_points: (M, 3) array
        downsampled_colors: (M, 3) array
    """
    if len(points) <= max_points:
        return points, colors
    
    indices = np.random.choice(len(points), max_points, replace=False)
    return points[indices], colors[indices]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, default='logs/tmp_save_np_small.npz', help='path to npz file')
    parser.add_argument("--indices", type=int, default=None, nargs='+', help='only load these frames for visualization')
    parser.add_argument("--sample_num", type=int, default=None, help='only sample several frames for visualization')
    parser.add_argument("--point_size", type=float, default=0.002, help='point size')  
    parser.add_argument("--scale_factor", type=float, default=8.0, help='point cloud scale factor for visualization')
    parser.add_argument("--port", type=int, default=7891, help='port')
    parser.add_argument("--record_width", type=int, default=1920, help='recording video width')
    parser.add_argument("--record_height", type=int, default=1280, help='recording video height')
    parser.add_argument("--camera_distance_factor", type=float, default=1.5, help='camera distance factor (smaller = closer, default: 0.3)')
    parser.add_argument("--max_points_per_frame", type=int, default=-1, help='maximum points per frame after downsampling (0 = disabled)')
    args = parser.parse_args()

    # Load data from npz file
    data = np.load(args.data_path, allow_pickle=True)
    
    
    point_cloud_data = [pc for pc in data.values()] #* a list of point clouds, every item is a point cloud, shape: [num_pc, 6]
    
    # Extract positions (first 3 dims) and colors (last 3 dims) from each point cloud
    # Each point cloud may have different number of points (N)
    positions = [pc[:, :3] for pc in point_cloud_data]  # list of [num_pc_i, 3] arrays
    colors = [pc[:, 3:6] for pc in point_cloud_data]   # list of [num_pc_i, 3] arrays
    
    num_frames = len(positions)
    logger.info(f"Loaded {num_frames} point clouds")
    # logger.info(f"Point cloud sizes: {[pc.shape[0] for pc in positions]}")

    if args.indices:
        indices = np.array(args.indices, dtype=np.int32)
    elif args.sample_num:
        indices = np.linspace(0, num_frames-1, args.sample_num)
        indices = np.round(indices).astype(np.int32)
    else:
        indices = np.array(list(range(num_frames)))
    
    # Select frames by indices (keep as list since sizes differ)
    positions = [positions[i] for i in indices]  # list of [num_pc_i, 3] arrays
    colors = [colors[i] for i in indices]        # list of [num_pc_i, 3] arrays
    
    # Normalize colors to [0, 255] if needed (assuming they might be in [0, 1] or [0, 255])
    # Check max value across all point clouds
    max_color = max([c.max() for c in colors])
    if max_color <= 1.0:
        colors = [(c * 255).astype(np.uint8) for c in colors]
    else:
        colors = [c.astype(np.uint8) for c in colors]

    # Apply downsampling if requested
    if args.max_points_per_frame > 0:
        logger.info("Applying downsampling to point clouds...")
        total_points_before = sum(len(p) for p in positions)
        downsampled_positions = []
        downsampled_colors = []
        
        for i, (pos, col) in enumerate(tqdm(zip(positions, colors), desc="Downsampling", total=len(positions))):
            # Apply random downsampling if needed
            if len(pos) > args.max_points_per_frame:
                pos, col = downsample_point_cloud_random(pos, col, args.max_points_per_frame)
            
            downsampled_positions.append(pos)
            downsampled_colors.append(col)
        
        positions = downsampled_positions
        colors = downsampled_colors
        
        total_points_after = sum(len(p) for p in positions)
        reduction_ratio = total_points_after / total_points_before if total_points_before > 0 else 1.0
        logger.info(f"Downsampling complete: {total_points_before} -> {total_points_after} points ({reduction_ratio*100:.1f}% retained)")
        logger.info(f"Average points per frame: {total_points_after / len(positions):.0f}")

    server = viser.ViserServer(port=args.port)
    server.request_share_url()
    num_frames = len(positions)

    # Calculate point cloud center (average across all frames)
    # Concatenate all positions from all frames (each frame may have different number of points)
    # 不用concat, 直接逐帧求mean/max/min，再整体求mean/max/min
    pc_center = np.mean([np.mean(pos, axis=0) for pos in positions], axis=0) * args.scale_factor
    logger.info(f"Point cloud center: {pc_center}")

    pc_min = np.min([np.min(pos, axis=0) for pos in positions], axis=0) * args.scale_factor
    pc_max = np.max([np.max(pos, axis=0) for pos in positions], axis=0) * args.scale_factor
    pc_size = np.max(pc_max - pc_min)
    camera_distance = pc_size * args.camera_distance_factor  # Place camera at specified distance factor

    # Calculate initial camera pose
    # Place camera in front of the scene (along +X axis) looking at center
    # You can change the direction by modifying the array [camera_distance, 0, 0]
    # Options:
    #   [camera_distance, 0, 0]  - Front (+X direction)
    #   [-camera_distance, 0, 0] - Back (-X direction)
    #   [0, camera_distance, 0]  - Right (+Y direction)
    #   [0, -camera_distance, 0] - Left (-Y direction)
    #   [0, 0, camera_distance]  - Top (+Z direction, current default)
    initial_camera_position = pc_center + np.array([camera_distance, camera_distance, -0.5 * camera_distance])  # Front of scene
    initial_camera_wxyz = compute_camera_pose(initial_camera_position, pc_center)
    logger.info(f"Initial camera position: {initial_camera_position}")
    logger.info(f"Initial camera orientation (wxyz): {initial_camera_wxyz}")
    
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
        
        
    
    # Add recording UI.
    with server.gui.add_folder("Recording"):
        gui_use_current_view = server.gui.add_checkbox("Use Current Browser View", False)
        gui_record_button = server.gui.add_button("Start Recording", disabled=False)
        gui_recording_progress = server.gui.add_text(
            "Ready to record",
            initial_value="Ready to record",
        )
    
    # Recording state (defined before handlers to avoid nonlocal issues)
    recording_state = {
        'is_recording': False,
        'record_width': args.record_width,
        'record_height': args.record_height,
        'camera_position': initial_camera_position,
        'camera_wxyz': initial_camera_wxyz,
    }
    
    def get_client_handle():
        """Get the first connected client handle for screenshot capture"""
        clients = list(server.get_clients().values())
        if len(clients) > 0:
            return clients[0]
        return None

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


    # Use a mutable container to store prev_timestep to avoid nonlocal issues
    prev_timestep_container = {'value': gui_timestep.value}

    # Add transform controls to allow manual dragging and rotating of camera
    camera_transform_controls = server.scene.add_transform_controls(
        name="/camera/recording_camera/controls",
        position=initial_camera_position,
        wxyz=initial_camera_wxyz,
        scale=camera_distance * 0.2,
        disable_rotations=False,  # Allow rotation
        visible=True
    )
    
    
    # Update camera position and orientation when transform controls are moved
    @camera_transform_controls.on_update
    def _(_) -> None:
        """Update camera position and orientation when transform controls are dragged/rotated"""
        new_position = np.array(camera_transform_controls.position)
        new_wxyz = np.array(camera_transform_controls.wxyz)
        
        # Update recording state
        recording_state['camera_position'] = new_position
        recording_state['camera_wxyz'] = new_wxyz

        
        logger.debug(f"Camera position updated: {new_position}, orientation: {new_wxyz}")
    
    logger.info("Camera visualization added to scene")
    logger.info("You can drag the camera position using the transform controls")
    
    # Load in frames - optimized: use single point cloud and update data instead of toggling visibility
    # Combine rotations: first rotate 90° around X-axis, then 180° around Z-axis
    rotation_x = tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0]))  # Rotate 90° around X-axis
    rotation_z = tf.SO3.exp(np.array([0.0, 0.0, np.pi]))        # Rotate 180° around Z-axis
    # Combine rotations by multiplying rotation matrices
    combined_matrix = rotation_x.as_matrix() @ rotation_z.as_matrix()
    combined_rotation = tf.SO3.from_matrix(combined_matrix)
    
    server.scene.add_frame(
        "/frames/point_cloud_frame",
        wxyz=combined_rotation.wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    
    # Pre-scale all positions and prepare colors for faster switching
    # Ensure all arrays are C-contiguous and in optimal data types for faster updates
    logger.info("Preparing point cloud data...")
    scaled_positions = []
    for pos in tqdm(positions, desc="Scaling positions"):
        # Scale and ensure contiguous float32 array for faster transfer
        scaled_pos = np.ascontiguousarray(pos * args.scale_factor, dtype=np.float32)
        scaled_positions.append(scaled_pos)
    
    # Ensure colors are contiguous uint8 arrays
    colors = [np.ascontiguousarray(c, dtype=np.uint8) for c in colors]
    
    scaled_point_size = args.point_size * args.scale_factor
    
    # Create a single point cloud object that we'll update instead of toggling visibility
    # This is much faster than switching visibility of multiple large point clouds
    initial_position = scaled_positions[0]
    initial_color = colors[0]
    
    point_cloud_handle = server.scene.add_point_cloud(
        name="/frames/point_cloud_frame/point_cloud",
        points=initial_position,
        colors=initial_color,
        point_size=scaled_point_size,
        point_shape="rounded",
    )
    
    # Store frame data for quick access
    frame_data = {
        'positions': scaled_positions,
        'colors': colors
    }

    # Update point cloud data when the timestep slider changes - much faster than toggling visibility
    # This callback is defined after point_cloud_handle and frame_data are created
    @gui_timestep.on_update
    def _(_) -> None:
        current_timestep = gui_timestep.value
        prev_timestep = prev_timestep_container['value']
        
        # Skip if same frame
        if current_timestep == prev_timestep:
            return
        
        # Optimized: update point cloud data directly
        # Data is already C-contiguous from preprocessing, so direct assignment is fastest
        # Direct assignment without atomic context for faster updates (non-blocking)
        point_cloud_handle.points = frame_data['positions'][current_timestep]
        point_cloud_handle.colors = frame_data['colors'][current_timestep]
        
        prev_timestep_container['value'] = current_timestep
    
    # Recording button handler - one-click recording in background thread
    # This is placed after frame_data and point_cloud_handle are defined so we can access them
    @gui_record_button.on_click
    def _(_) -> None:
        # Check if a client is connected for recording
        client_handle = get_client_handle()
        if client_handle is None:
            logger.error("Cannot start recording: No client connected")
            logger.error("Please open the viser viewer in a browser first")
            return
        
        # Check if already recording
        if recording_state['is_recording']:
            logger.warning("Recording already in progress")
            return
        
        # Disable button during recording
        gui_record_button.disabled = True
        gui_record_button.name = "Recording..."
        
        # Hide camera transform controls during recording
        camera_transform_controls.visible = False
        
        # Get recording parameters
        use_current_view = gui_use_current_view.value
        camera_position = recording_state['camera_position'].copy()
        camera_wxyz = recording_state['camera_wxyz'].copy()
        record_width = recording_state['record_width']
        record_height = recording_state['record_height']
        
        logger.info("Recording started - will play all frames and save video in background thread")
        logger.info(f"Total frames to record: {num_frames}")
        if use_current_view:
            logger.info("Using current browser camera view for recording")
        else:
            logger.info(f"Camera set to position: {camera_position}")
            logger.info(f"Camera orientation (wxyz): {camera_wxyz}")
        
        def recording_thread():
            """Background thread that handles all recording: play, render, and save"""
            try:
                recording_state['is_recording'] = True
                frames = []
                
                # Reset to first frame
                gui_timestep.value = 0
                gui_recording_progress.value = f"Recording: 0/{num_frames} frames (0%)"
                
                # Loop through all frames
                for frame_idx in range(num_frames):
                    # Update point cloud to current frame
                    gui_timestep.value = frame_idx
                    point_cloud_handle.points = frame_data['positions'][frame_idx]
                    point_cloud_handle.colors = frame_data['colors'][frame_idx]
                    
                    # Capture frame (get_render is synchronous, waits for browser to complete)
                    try:
                        if use_current_view:
                            render_image = client_handle.get_render(
                                height=record_height,
                                width=record_width,
                                transport_format='jpeg',
                            )
                        else:
                            render_image = client_handle.get_render(
                                height=record_height,
                                width=record_width,
                                position=camera_position,
                                wxyz=camera_wxyz,
                                transport_format='jpeg'
                            )
                        
                        # Convert to PIL Image and store
                        img = Image.fromarray(render_image.astype(np.uint8))
                        frames.append(img)
                        
                    except Exception as e:
                        logger.warning(f"Frame {frame_idx} capture failed: {e}")
                    
                    # Update progress
                    progress_percent = ((frame_idx + 1) / num_frames) * 100
                    gui_recording_progress.value = f"Recording: {frame_idx + 1}/{num_frames} frames ({progress_percent:.1f}%)"
                    
                    if (frame_idx + 1) % 10 == 0 or frame_idx == num_frames - 1:
                        logger.info(f"Recording progress: {frame_idx + 1}/{num_frames} frames ({progress_percent:.1f}%)")
                
                # Save video
                gui_recording_progress.value = f"Recording completed: {len(frames)}/{num_frames} frames captured. Saving video..."
                logger.info(f"Recording completed. Captured {len(frames)} frames")
                
                if len(frames) > 0:
                    output_dir = os.path.dirname(args.data_path)
                    output_path = os.path.join(output_dir, 'pc_video.mp4')
                    
                    # Convert frames to numpy arrays
                    frames_to_save = [np.array(f) for f in frames]
                    
                    try:
                        save_start_time = time.time()
                        
                        writer = imageio.get_writer(output_path, fps=30)
                        for frame in tqdm(frames_to_save, desc="Saving video"):
                            writer.append_data(frame)
                        writer.close()
                        
                        save_elapsed_time = time.time() - save_start_time
                        logger.info(f"Video saved to {output_path}")
                        logger.info(f"Video saving took {save_elapsed_time:.2f} seconds ({save_elapsed_time/60:.2f} minutes)")
                        gui_recording_progress.value = f"Video saved: {output_path} (took {save_elapsed_time:.2f}s)"
                    except Exception as e:
                        logger.error(f"Failed to save video: {e}")
                        gui_recording_progress.value = f"Error saving video: {e}"
                
            except Exception as e:
                logger.error(f"Recording thread error: {e}")
                gui_recording_progress.value = f"Recording error: {e}"
            finally:
                # Cleanup: re-enable button and show camera transform controls
                recording_state['is_recording'] = False
                gui_record_button.disabled = False
                gui_record_button.name = "Start Recording"
                camera_transform_controls.visible = True
        
        # Start recording in background thread
        thread = threading.Thread(target=recording_thread, daemon=True)
        thread.start()

    # Playback update loop - simplified, recording is now handled in background thread
    while True:
        if gui_playing.value and not recording_state['is_recording']:
            # Only advance frames when playing manually (not during recording)
            current_frame = gui_timestep.value
            next_frame = (current_frame + 1) % num_frames
            gui_timestep.value = next_frame
        
        # Sleep to control playback speed (30 fps)
        # time.sleep(1.0 / 30)
        
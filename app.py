
import os
import gradio as gr


import numpy as np
import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm
from tools.common_utils import save_video
from dkt.pipelines.pipeline import DKTPipeline, ModelConfig


import cv2
import copy
import trimesh

from os.path import join
from tools.depth2pcd import depth2pcd
# from moge.model.v2 import MoGeModel


from tools.eval_utils import transfer_pred_disp2depth, colorize_depth_map
import glob
import datetime
import shutil
import tempfile
import spaces
import time


#* better for bg:  logs/outs/train/remote/sft-T2SQNet_glassverse_cleargrasp_HISS_DREDS_DREDS_glassverse_interiorverse-4gpus-origin-lora128-1.3B-rgb_depth-w832-h480-Wan2.1-Fun-Control-2025-10-28-23:26:41/epoch-0-20000.safetensors
PROMPT = 'depth'
NEGATIVE_PROMPT = ''

height = 480
width = 832
window_size = 21
DKT_PIPELINE = DKTPipeline()

example_inputs = [
    ["examples/1.mp4", "1.3B", 5, 3],
    ["examples/33.mp4", "1.3B", 5, 3],
    ["examples/7.mp4", "1.3B", 5, 3],
    ["examples/8.mp4", "1.3B", 5, 3],
    ["examples/9.mp4", "1.3B", 5, 3],
    ["examples/36.mp4", "1.3B", 5, 3],
    ["examples/39.mp4", "1.3B", 5, 3],
    ["examples/10.mp4", "1.3B", 5, 3],
    ["examples/30.mp4", "1.3B", 5, 3],
    ["examples/3.mp4", "1.3B", 5, 3],
    ["examples/32.mp4", "1.3B", 5, 3],
    ["examples/35.mp4", "1.3B", 5, 3],
    ["examples/40.mp4", "1.3B", 5, 3],
    ["examples/2.mp4", "1.3B", 5, 3],
]





def pmap_to_glb(point_map, valid_mask, frame) -> trimesh.Scene:
    pts_3d = point_map[valid_mask] * np.array([-1, -1, 1])
    pts_rgb = frame[valid_mask] 

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(
        vertices=pts_3d, colors=pts_rgb
    )
    
    scene_3d.add_geometry(point_cloud_data)
    return scene_3d



def create_simple_glb_from_pointcloud(points, colors, glb_filename):
    try:
        if len(points) == 0:
            logger.warning(f"No valid points to create GLB for {glb_filename}")
            return False
        
        if colors is not None:
            # logger.info(f"Adding colors to GLB: shape={colors.shape}, range=[{colors.min():.3f}, {colors.max():.3f}]")
            pts_rgb = colors
        else:
            logger.info("No colors provided, adding default white colors")
            pts_rgb = np.ones((len(points), 3))
        
        valid_mask = np.ones(len(points), dtype=bool)
        
        scene_3d = pmap_to_glb(points, valid_mask, pts_rgb)
        
        scene_3d.export(glb_filename)
        # logger.info(f"Saved GLB file using trimesh: {glb_filename}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating GLB from pointcloud using trimesh: {str(e)}")
        return False






def process_video(
    video_file,
    model_size,
    num_inference_steps,
    overlap
):
    global height
    global width
    global window_size
    global DKT_PIPELINE


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cur_save_dir = tempfile.mkdtemp(prefix=f'dkt_{timestamp}_{model_size}_')
    


    
    start_time = time.time()

    prediction_result = DKT_PIPELINE(
        video_file,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        overlap=overlap,
        return_rgb=True
    )

    end_time = time.time()
    spend_time = end_time - start_time
    logger.info(f"DKT_PIPELINE spend time: {spend_time:.2f} seconds for depth prediction")
    print(f"DKT_PIPELINE spend time: {spend_time:.2f} seconds for depth prediction")

    
    

    frame_length = len(prediction_result['rgb_frames'])
    vis_pc_num = 4
    indices = np.linspace(0, frame_length-1, vis_pc_num)
    indices = np.round(indices).astype(np.int32)
    
  
    pc_start_time = time.time()
    pcds = DKT_PIPELINE.prediction2pc_v2(prediction_result['depth_map'], prediction_result['rgb_frames'], indices, return_pcd=True)
    pc_end_time = time.time()
    pc_spend_time = pc_end_time - pc_start_time
    logger.info(f"prediction2pc_v2 spend time: {pc_spend_time:.2f} seconds for point cloud extraction")
    print(f"prediction2pc_v2 spend time: {pc_spend_time:.2f} seconds for point cloud extraction")

    glb_files = []

    for idx, pcd in enumerate(pcds):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        logger.info(f'points:{points.shape}, colors: {colors.shape}')
        print(f'points:{points.shape}, colors: {colors.shape}')


        
        points[:, 2] = -points[:, 2]  
        points[:, 0] = -points[:, 0]  
        

        glb_filename = os.path.join(cur_save_dir, f'{timestamp}_{idx:02d}.glb')
        success = create_simple_glb_from_pointcloud(points, colors, glb_filename)
        if not success:
            logger.warning(f"Failed to save GLB file: {glb_filename}")
            print(f"Failed to save GLB file: {glb_filename}")

        glb_files.append(glb_filename)

    
    
    
    #* save depth predictions video
    output_filename = f"output_{timestamp}.mp4"
    output_path = os.path.join(cur_save_dir, output_filename)

    
    cap = cv2.VideoCapture(video_file)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    save_video(prediction_result['colored_depth_map'], output_path, fps=input_fps, quality=8)
    return output_path, glb_files






#* gradio creation and initialization


css = """
 #download {
                height: 118px;
            }
            .slider .inner {
                width: 5px;
                background: #FFF;
            }
            .viewport {
                aspect-ratio: 4/3;
            }
            .tabs button.selected {
                font-size: 20px !important;
                color: crimson !important;
            }
            h1 {
                text-align: center;
                display: block;
            }
            h2 {
                text-align: center;
                display: block;
            }
            h3 {
                text-align: center;
                display: block;
            }
            .md_feedback li {
                margin-bottom: 0px !important;
            }
"""



head_html = """
            <script async src="https://www.googletagmanager.com/gtag/js?id=G-1FWSVCGZTG"></script>
            <script>
                window.dataLayer = window.dataLayer || [];
                function gtag() {dataLayer.push(arguments);}
                gtag('js', new Date());
                gtag('config', 'G-1FWSVCGZTG');
            </script>
"""




with gr.Blocks(css=css, title="DKT", head=head_html) as demo:
    # gr.Markdown(title, elem_classes=["title"])
    gr.Markdown(
        """
        # Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation
        <p align="center">
        <a title="Website" href="https://daniellli.github.io/projects/DKT/" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            <img src="https://www.obukhov.ai/img/badges/badge-website.svg">
        </a>
        <a title="Github" href="https://github.com/Daniellli/DKT" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            <img src="https://img.shields.io/github/stars/Daniellli/DKT?style=social" alt="badge-github-stars">
        </a>
        <a title="Social" href="https://x.com/xshocng1" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
            <img src="https://www.obukhov.ai/img/badges/badge-social.svg" alt="social">
        </a>
    """
    )
    # gr.Markdown(description, elem_classes=["description"])
    # gr.Markdown("### Video Processing Demo", elem_classes=["description"])

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video", elem_id='video-display-input')
            
            model_size = gr.Radio(
                # choices=["1.3B", "14B"],
                choices=["1.3B"],
                value="1.3B",
                label="Model Size"
            )


            with gr.Accordion("Advanced Parameters", open=False):
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=50, value=5, step=1,
                    label="Number of Inference Steps"
                )
                overlap = gr.Slider(
                    minimum=1, maximum=20, value=3, step=1,
                    label="Overlap"
                )
                
            submit = gr.Button(value="Compute Depth", variant="primary")
        
        with gr.Column():
            output_video = gr.Video(
                label="Depth Outputs", 
                elem_id='video-display-output',
                autoplay=True
            )
            vis_video = gr.Video(
                label="Visualization Video", 
                visible=False,
                autoplay=True
            )

    with gr.Row():
        gr.Markdown("### 3D Point Cloud Visualization", elem_classes=["title"])
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            output_point_map0 = gr.Model3D(
                label="Point Cloud Key Frame 1",
                clear_color=[1.0, 1.0, 1.0, 1.0],
                interactive=False,
            )
        with gr.Column(scale=1):
            output_point_map1 = gr.Model3D(
                label="Point Cloud Key Frame 2",
                clear_color=[1.0, 1.0, 1.0, 1.0],
                interactive=False
            )
        
    
    with gr.Row(equal_height=True):
        
        with gr.Column(scale=1):
            output_point_map2 = gr.Model3D(
                label="Point Cloud Key Frame 3",
                clear_color=[1.0, 1.0, 1.0, 1.0],
                interactive=False
            )
        with gr.Column(scale=1):
            output_point_map3 = gr.Model3D(
                label="Point Cloud Key Frame 4",
                clear_color=[1.0, 1.0, 1.0, 1.0],
                interactive=False
            )

    def on_submit(video_file, model_size, num_inference_steps, overlap):
        logger.info('on_submit is calling')
        if video_file is None:
            return None, None, None, None, None, None, "Please upload a video file"
        
        try:
            
            start_time = time.time()
            output_path, glb_files = process_video(
                video_file, model_size, num_inference_steps, overlap
            )
            spend_time = time.time() - start_time
            logger.info(f"Total spend time in on_submit: {spend_time:.2f} seconds")
            print(f"Total spend time in on_submit: {spend_time:.2f} seconds")

            
            if output_path is None:
                return None, None, None, None, None, None, glb_files
            
            model3d_outputs = [None] * 4
            if glb_files:
                for i, glb_file in enumerate(glb_files[:4]):
                    if os.path.exists(glb_file):
                        model3d_outputs[i] = glb_file
            


            return output_path, None, *model3d_outputs
                
        except Exception as e:
            logger.error(e)
            return None, None, None, None, None, None

    
    submit.click(
        on_submit,
        inputs=[
            input_video, model_size, num_inference_steps, overlap
        ],
        outputs=[
            output_video, vis_video, output_point_map0, output_point_map1, output_point_map2, output_point_map3
        ]
    )
    

    
    logger.info(f'there are {len(example_inputs)} demo files')
    print(f'there are {len(example_inputs)} demo files')
        
    examples = gr.Examples(
        examples=example_inputs, 
        inputs=[input_video, model_size, num_inference_steps, overlap], 
        outputs=[
            output_video, vis_video, 
            output_point_map0, output_point_map1, output_point_map2, output_point_map3
        ], 
        fn=on_submit,
        examples_per_page=12,
        cache_examples=False
    )


if __name__ == '__main__':
    
    #* main code, model and moge model initialization
    demo.queue().launch(share = True)
    
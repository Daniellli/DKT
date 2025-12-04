
import os
import gradio as gr
import numpy as np
import torch
from PIL import Image
from loguru import logger
from tqdm import tqdm
from tools.common_utils import save_video
from dkt.pipelines.pipelines import WanVideoPipeline, ModelConfig

import cv2
import copy
import trimesh
from gradio_litmodel3d import LitModel3D
from os.path import join
from tools.depth2pcd import depth2pcd
from moge.model.v2 import MoGeModel
from tools.eval_utils import transfer_pred_disp2depth, colorize_depth_map
import glob
import datetime
import shutil
import tempfile
from dkt.pipelines.pipelines import extract_frames_from_video_file, resize_frame

PIPE_1_3B = None
MOGE_MODULE =  None
PROMPT = 'depth'
NEGATIVE_PROMPT = ''

SAVE_DIR = "logs/gradio"
os.makedirs(SAVE_DIR, exist_ok=True)



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








def load_moge_model(device="cuda:0"):
    global MOGE_MODULE
    if MOGE_MODULE is not None:
        return MOGE_MODULE
    logger.info(f"Loading MoGe model on {device}...")
    cached_model_path = 'checkpoints/moge_ckpt/moge-2-vitl-normal/model.pt'

    if os.path.exists(cached_model_path):
        logger.info(f"Found cached model at {cached_model_path}, loading from cache...")
        MOGE_MODULE = MoGeModel.from_pretrained(cached_model_path).to(device)
    else:
        logger.info(f"Cache not found at {cached_model_path}, downloading from HuggingFace...")
        os.makedirs(os.path.dirname(cached_model_path), exist_ok=True)
        MOGE_MODULE = MoGeModel.from_pretrained('Ruicheng/moge-2-vitl-normal', cache_dir=os.path.dirname(cached_model_path)).to(device)
    
    return MOGE_MODULE


def load_model_1_3b(device="cuda:0"):
    global PIPE_1_3B
    
    if PIPE_1_3B is not None:
        return PIPE_1_3B
    
    logger.info(f"Loading 1.3B model on {device}...")
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-1.3B-Control",
                origin_file_pattern="diffusion_pytorch_model*.safetensors",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-1.3B-Control",
                origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-1.3B-Control",
                origin_file_pattern="Wan2.1_VAE.pth",
                offload_device="cpu",
            ),
            ModelConfig(
                model_id="PAI/Wan2.1-Fun-1.3B-Control",
                origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
                offload_device="cpu",
            ),
        ],
        training_strategy="origin",
    )
    
    
    lora_config = ModelConfig(
        model_id="Daniellesry/DKT-Depth-1-3B",
        origin_file_pattern="dkt-1-3B.safetensors",
        offload_device="cpu",
    )
    lora_config.download_if_necessary(use_usp=False)
    
    pipe.load_lora(pipe.dit, lora_config.path, alpha=1.0)#todo is it work?
    pipe.enable_vram_management()

    
    PIPE_1_3B = pipe
    
    return pipe





    

def get_model(model_size):
    if model_size == "1.3B":
        assert PIPE_1_3B is not None, "1.3B model not initialized"
        return PIPE_1_3B
    else:
        raise ValueError(f"Unsupported model size: {model_size}")



def process_video(
    video_file,
    model_size,
    height,
    width,
    num_inference_steps,
    window_size,
    overlap
):
    try:
        pipe = get_model(model_size)
        if pipe is None:
            return None, f"Model {model_size} not initialized. Please restart the application."
        
        
        tmp_video_path = video_file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cur_save_dir = join(SAVE_DIR, timestamp+f'_'+model_size)
        os.makedirs(cur_save_dir, exist_ok=True)


        
        original_filename = f"input_{timestamp}.mp4"
        dst_path = os.path.join(cur_save_dir, original_filename)
        shutil.copy2(tmp_video_path, dst_path)
        origin_frames, input_fps = extract_frames_from_video_file(tmp_video_path)
        
        if not origin_frames:
            return None, "Failed to extract frames from video"
        
        logger.info(f"Extracted {len(origin_frames)} frames from video")

        
        original_width, original_height = origin_frames[0].size
        ROTATE = False 
        if original_width <  original_height:
            ROTATE = True
            origin_frames = [x.transpose(Image.ROTATE_90) for x in origin_frames]
            tmp = original_width
            original_width = original_height
            original_height = tmp
        
        frames = [resize_frame(frame, height, width) for frame in origin_frames]
        frame_length = len(frames)
        if (frame_length - 1) % 4 != 0:
            new_len = ((frame_length - 1) // 4 + 1) * 4 + 1
            frames = frames + [copy.deepcopy(frames[-1]) for _ in range(new_len - frame_length)]

        
        control_video = frames
        video, vae_outs = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            control_video=control_video,
            height=height,
            width=width,
            num_frames=len(control_video),
            seed=1,
            tiled=False,
            num_inference_steps=num_inference_steps,
            sliding_window_size=window_size,
            sliding_window_stride=window_size - overlap,
            cfg_scale=1.0,
        )

        #* moge process 
        torch.cuda.empty_cache()
        processed_video = video[:frame_length]
        processed_video = [resize_frame(frame, original_height, original_width) for frame in processed_video]

        if ROTATE:
            processed_video = [x.transpose(Image.ROTATE_270) for x in processed_video]
            origin_frames = [x.transpose(Image.ROTATE_270) for x in origin_frames]
            

        output_filename = f"output_{timestamp}.mp4"
        output_path = os.path.join(cur_save_dir, output_filename)
        color_predictions = []
        if PROMPT == 'depth':
            prediced_depth_map_np = [np.array(item).astype(np.float32).mean(-1)  for item in processed_video]
            prediced_depth_map_np = np.stack(prediced_depth_map_np)
            prediced_depth_map_np = prediced_depth_map_np/ 255.0
            __min = prediced_depth_map_np.min()
            __max = prediced_depth_map_np.max()
            prediced_depth_map_np = (prediced_depth_map_np - __min) / (__max - __min)
            color_predictions = [colorize_depth_map(item) for item in prediced_depth_map_np]
        else:
            color_predictions = processed_video
        save_video(color_predictions, output_path, fps=input_fps, quality=5)


        
        frame_num = len(origin_frames)
        resize_W,resize_H = origin_frames[0].size
        
        vis_pc_num = 4
        indices = np.linspace(0, frame_num-1, vis_pc_num)
        indices = np.round(indices).astype(np.int32)
        pc_save_dir = os.path.join(cur_save_dir, 'pointclouds')
        os.makedirs(pc_save_dir, exist_ok=True)

        glb_files = []
        moge_device = MOGE_MODULE.device if MOGE_MODULE is not None else torch.device("cuda:0")
        
        for idx in tqdm(indices):
            orgin_rgb_frame = origin_frames[idx]
            predicted_depth = processed_video[idx]

            # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
            input_image_np = np.array(orgin_rgb_frame)  # Convert PIL Image to numpy array
            input_image = torch.tensor(input_image_np / 255, dtype=torch.float32, device=moge_device).permute(2, 0, 1) 

            output = MOGE_MODULE.infer(input_image)
            #* "dict_keys(['points', 'intrinsics', 'depth', 'mask', 'normal'])"
            moge_intrinsics = output['intrinsics'].cpu().numpy()
            moge_mask = output['mask'].cpu().numpy()
            moge_depth = output['depth'].cpu().numpy()
        
            predicted_depth = np.array(predicted_depth)
            predicted_depth = predicted_depth.mean(-1) / 255.0
    
            metric_depth = transfer_pred_disp2depth(predicted_depth, moge_depth, moge_mask)
            
            moge_intrinsics[0, 0] *= resize_W 
            moge_intrinsics[1, 1] *= resize_H
            moge_intrinsics[0, 2] *= resize_W
            moge_intrinsics[1, 2] *= resize_H
            
            # pcd = depth2pcd(metric_depth, moge_intrinsics, color=cv2.cvtColor(input_image_np, cv2.COLOR_BGR2RGB), input_mask=moge_mask, ret_pcd=True)
            pcd = depth2pcd(metric_depth, moge_intrinsics, color=input_image_np, input_mask=moge_mask, ret_pcd=True)
            
            # pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) * np.array([1, -1, -1], dtype=np.float32))
            
            apply_filter = True
            if apply_filter:
                cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=3.0)
                pcd = pcd.select_by_index(ind)

            #* save pcd:  o3d.io.write_point_cloud(f'{pc_save_dir}/{timestamp}_{idx:02d}.ply', pcd)
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else None
            glb_filename = os.path.join(pc_save_dir, f'{timestamp}_{idx:02d}.glb')
            success = create_simple_glb_from_pointcloud(points, colors, glb_filename)
            if not success:
                logger.warning(f"Failed to save GLB file: {glb_filename}")

            glb_files.append(glb_filename)

        return output_path, glb_files
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return None, f"Error: {str(e)}"




def main():

    #* gradio creation and initialization


    css = """
    #video-display-container {
        max-height: 100vh;
    }
    #video-display-input {
        max-height: 80vh;
    }
    #video-display-output {
        max-height: 80vh;
    }
    #download {
        height: 62px;
    }
    .title {
        text-align: center;
    }
    .description {
        text-align: center;
    }
    .gradio-examples {
        max-height: 400px;
        overflow-y: auto;
    }
    .gradio-examples .examples-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 10px;
        padding: 10px;
    }
    .gradio-container .gradio-examples .pagination,
    .gradio-container .gradio-examples .pagination button,
    div[data-testid="examples"] .pagination,
    div[data-testid="examples"] .pagination button {
        font-size: 28px !important;
        font-weight: bold !important;
        padding: 15px 20px !important;
        min-width: 60px !important;
        height: 60px !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
        margin: 8px !important;
        display: inline-block !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="examples"] .pagination button:not(.active),
    .gradio-container .gradio-examples .pagination button:not(.active) {
        font-size: 32px !important;
        font-weight: bold !important;
        padding: 15px 20px !important;
        min-width: 60px !important;
        height: 60px !important;
        background: linear-gradient(135deg, #8a9cf0 0%, #9a6bb2 100%) !important;
        opacity: 0.8 !important;
    }

    div[data-testid="examples"] .pagination button:hover,
    .gradio-container .gradio-examples .pagination button:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
        opacity: 1 !important;
    }

    div[data-testid="examples"] .pagination button.active,
    .gradio-container .gradio-examples .pagination button.active {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        box-shadow: 0 4px 8px rgba(17,153,142,0.4) !important;
        opacity: 1 !important;
    }

    button[class*="pagination"],
    button[class*="page"] {
        font-size: 28px !important;
        font-weight: bold !important;
        padding: 15px 20px !important;
        min-width: 60px !important;
        height: 60px !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        cursor: pointer !important;
        margin: 8px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
        transition: all 0.3s ease !important;
    }
    """



    head_html = """
    <link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🦾%3C/text%3E%3C/svg%3E">
    <link rel="shortcut icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🦾%3C/text%3E%3C/svg%3E">
    <link rel="icon" type="image/png" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E🦾%3C/text%3E%3C/svg%3E">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    """


    height = 480
    width = 832
    window_size = 21
    with gr.Blocks(css=css, title="DKT", head=head_html) as demo:
        # gr.Markdown( "# Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation", elem_classes=["title"])
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
                output_point_map0 = LitModel3D(
                    label="Point Cloud Key Frame 1",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False,
                    # height=400,
                    
                )
            with gr.Column(scale=1):
                output_point_map1 = LitModel3D(
                    label="Point Cloud Key Frame 2",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )
            
        
        with gr.Row(equal_height=True):
            
            with gr.Column(scale=1):
                output_point_map2 = LitModel3D(
                    label="Point Cloud Key Frame 3",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )
            with gr.Column(scale=1):
                output_point_map3 = LitModel3D(
                    label="Point Cloud Key Frame 4",
                    clear_color=[1.0, 1.0, 1.0, 1.0],
                    interactive=False
                )

        def on_submit(video_file, model_size, num_inference_steps, overlap):
            if video_file is None:
                return None, None, None, None, None, None, "Please upload a video file"
            try:
                output_path, glb_files = process_video(
                    video_file, model_size, height, width, num_inference_steps, window_size, overlap
                )

                
                if output_path is None:
                    return None, None, None, None, None, None, glb_files
                
                model3d_outputs = [None] * 4
                if glb_files:
                    for i, glb_file in enumerate(glb_files[:4]):
                        if os.path.exists(glb_file):
                            model3d_outputs[i] = glb_file
                


                return output_path, None, *model3d_outputs
                    
            except Exception as e:
                return None, None, None, None, None, None, f"Error: {str(e)}"

        
        submit.click(
            on_submit,
            inputs=[
                input_video, model_size, num_inference_steps, overlap
            ],
            outputs=[
                output_video, vis_video, 
                output_point_map0, output_point_map1, output_point_map2, output_point_map3
            ]
        )
        
        
        
        example_files = glob.glob('examples/*')
        if example_files:
            example_inputs = []
            for file_path in example_files:
                example_inputs.append([file_path, "1.3B"])
            
            examples = gr.Examples(
                examples=example_inputs, 
                inputs=[input_video, model_size], 
                outputs=[
                    output_video, vis_video, 
                    output_point_map0, output_point_map1, output_point_map2, output_point_map3
                ], 
                fn=on_submit,
                examples_per_page=6
            )


    #* main code, model and moge model initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model_1_3b(device=device)
    load_moge_model(device=device)
    torch.cuda.empty_cache()
    demo.queue().launch(share = True,server_name="0.0.0.0", server_port=7860)


if __name__ == '__main__':
    main()

import os
import gradio as gr
import numpy as np
import torch
from PIL import Image
from loguru import logger
from dkt import load_state_dict
from tools.common_utils import save_video
from dkt.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
import cv2
import copy
import shutil
import glob
import datetime
from os.path import join
import time



SAVE_DIR = "logs/gradio_normal_mask"
os.makedirs(SAVE_DIR, exist_ok=True)
LORA = True
MASK_MODEL = None
NORMAL_MODEL = None
MASK_MODEL_PATH = "DKT_models/T2SQNet_Trans10K_cleargrasp_1.3B_mask_95k_lora.safetensors"
MASK_PROMPT = 'transparent object segmentation'
NEGATIVE_PROMPT = ''
NORMAL_MODEL_PATH = "DKT_models/T2SQNet_glassverse_cleargrasp_interiorverse_DREDS_DREDS_glassverse_14B_normal_60K_lora.safetensors"
NORMAL_PROMPT = 'normal'




def resize_frame(frame, height, width):
    frame = np.array(frame)
    frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    frame = torch.nn.functional.interpolate(frame, (height, width), mode="bicubic", align_corners=False, antialias=True)
    frame = (frame.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    frame = Image.fromarray(frame)
    return frame






def extract_frames_from_video_file(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 15.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = Image.fromarray(frame_rgb)
            frames.append(frame_rgb)
        
        cap.release()
        return frames, fps
    except Exception as e:
        logger.error(f"Error extracting frames from {video_path}: {str(e)}")
        return [], 15.0




def load_model_1_3b(model_path,device="cuda:0"):
    
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
    
    if LORA:
        new_weight_alpha = 1.0
        pipe.load_lora(pipe.dit, model_path, alpha=new_weight_alpha)
        logger.info(f"Loaded LoRA model from {model_path}, with alpha {new_weight_alpha}")
    else:
        pipe.dit.load_state_dict(load_state_dict(model_path))
        logger.info(f"Loaded full model from {model_path}")
    
    pipe.enable_vram_management()
    

    return pipe




def load_model_14b(model_path,device="cuda:1"):
    
    logger.info(f"Loading 14B model on {device}...")
    
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
            ModelConfig(model_id="PAI/Wan2.1-Fun-14B-Control", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
        ],
        redirect_common_files=False,
    )
    
    if LORA:
        new_weight_alpha = 1.0
        pipe.load_lora(pipe.dit, model_path, alpha=new_weight_alpha)
        logger.info(f"Loaded LoRA model from {model_path}, with alpha {new_weight_alpha}")
    else:
        pipe.dit.load_state_dict(load_state_dict(model_path))
        logger.info(f"Loaded full model from {model_path}")
    
    pipe.enable_vram_management()
    
    
    return pipe


    

def get_model(model_type):
    if model_type == "MASK_1.3B":
        assert MASK_MODEL is not None, "MASK_1.3B model not initialized"
        return MASK_MODEL, MASK_PROMPT
    elif model_type == "NORMAL_14B":
        assert NORMAL_MODEL is not None, "NORMAL_14B model not initialized"
        return NORMAL_MODEL, NORMAL_PROMPT
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def process_video(
    video_file,
    model_type,
    height,
    width,
    num_inference_steps,
    window_size,
    overlap
):
    try:
        pipe, prompt= get_model(model_type)
        if pipe is None:
            return None, f"Model {model_type} not initialized. Please restart the application."
        
        tmp_video_path = video_file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        cur_save_dir = join(SAVE_DIR, timestamp+f'_'+model_type)
        os.makedirs(cur_save_dir, exist_ok=True)
        
        
        original_filename = f"input_{timestamp}.mp4"
        dst_path = os.path.join(cur_save_dir, original_filename)
        shutil.copy2(tmp_video_path, dst_path)
        logger.info(f"Saved input video to: {dst_path}")
        
        origin_frames, input_fps = extract_frames_from_video_file(tmp_video_path)
        
        if not origin_frames:
            return None, "Failed to extract frames from video"
        
        logger.info(f"Extracted {len(origin_frames)} frames from video")
        
        
        original_height, original_width = origin_frames[0].size
                
        frames = [resize_frame(frame, height, width) for frame in origin_frames]
        frame_length = len(frames)
        if (frame_length - 1) % 4 != 0:
            new_len = ((frame_length - 1) // 4 + 1) * 4 + 1
            frames = frames + [copy.deepcopy(frames[-1]) for _ in range(new_len - frame_length)]
        
        control_video = frames
        video, __ = pipe(
            prompt=prompt,
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

        torch.cuda.empty_cache()
        processed_video = video[:frame_length] 
        processed_video = [resize_frame(frame, original_width, original_height) for frame in processed_video]

        
        output_path = os.path.join(cur_save_dir, f"output_{timestamp}.mp4")
        save_video(processed_video, output_path, fps=input_fps, quality=5)

        return output_path, None
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return None, f"Error: {str(e)}"

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


title = "# Diffusion Knows Transparency: Repurposing Video Diffusion for Transparent Object Depth and Normal Estimation "
# description = """Official demo for **DKT **."""

# with gr.Blocks(css=css, title="DKT - Diffusion Knows Transparency", favicon_path="favicon.ico") as demo:
height = 480
width = 832
window_size=21
with gr.Blocks(css=css, title="DKT", head=head_html) as demo:
    gr.Markdown(title, elem_classes=["title"])
    # gr.Markdown(description, elem_classes=["description"])
    # gr.Markdown("### Video Processing Demo", elem_classes=["description"])

    with gr.Row():
        with gr.Column():
            input_video = gr.Video(label="Input Video", elem_id='video-display-input')
            
            model_type = gr.Radio(
                choices=["MASK_1.3B", "NORMAL_14B"],
                value="MASK_1.3B",
                label="Model Type"
            )

            with gr.Accordion("Advanced Parameters", open=False):
                
                num_inference_steps = gr.Slider(
                    minimum=1, maximum=50, value=5, step=1,
                    label="Number of Inference Steps"
                )
                
                overlap = gr.Slider(
                    minimum=1, maximum=20, value=5, step=1,
                    label="Overlap"
                )
                
            submit = gr.Button(value="Compute", variant="primary")
        
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
    def on_submit(video_file, model_type, num_inference_steps, overlap):
        if video_file is None:
            return None, None, None, None, None, None, "Please upload a video file"
        
        try:
            import time
            start_time = time.time()
            output_path, glb_files = process_video(
                video_file, model_type, height, width, num_inference_steps, window_size, overlap
            )

            elapsed_time = time.time() - start_time
            print(f"process_video elapsed time: {elapsed_time:.2f}s, model_type: {model_type}")
            
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
            input_video, model_type, num_inference_steps, overlap
        ],
        outputs=[
            output_video, vis_video
        ]
    )
    
    example_files = glob.glob('logs/appendix/gradio/*')
    if example_files:
        example_inputs = []
        for file_path in example_files:
            example_inputs.append([file_path, "MASK_1.3B"])
        
        examples = gr.Examples(
            examples=example_inputs, 
            inputs=[input_video, model_type], 
            outputs=[
                output_video, vis_video
            ], 
            fn=on_submit,
            examples_per_page=6
        )

if __name__ == '__main__':
    gpu_count = torch.cuda.device_count()
    logger.info(f"Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        logger.warning(f"Only {gpu_count} GPU(s) available, both models will use GPU 0")
        device_1_3b = "cuda:0"
        device_14b = "cuda:0"
    else:
        device_1_3b = "cuda:0"
        device_14b = "cuda:1"
        logger.info(f"1.3B model will use {device_1_3b}, 14B model will use {device_14b}")

    
    MASK_MODEL = load_model_1_3b(model_path=MASK_MODEL_PATH,device=device_1_3b)
    NORMAL_MODEL = load_model_14b(model_path=NORMAL_MODEL_PATH,device=device_14b)
    logger.info("All models initialization completed!")
    torch.cuda.empty_cache()
    demo.queue().launch(share=False, server_name="0.0.0.0", server_port=7860)

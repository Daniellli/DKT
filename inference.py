import os





import os 
import numpy as np
import glob
from decord import cpu, VideoReader

# Set tokenizers parallelism to avoid warnings when forking processes
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dkt.pipelines.pipelines import DKTPipeline
from tools.common_utils import save_video


import argparse


from tqdm import tqdm
from os.path import join


parser = argparse.ArgumentParser()
parser.add_argument("--jsonl_path", type=str, default='logs/appendix/all_videos_1.jsonl')
parser.add_argument("--save_dir", type=str, default='logs/inference_results')

args = parser.parse_args()


def pcd2pc_array(pcd):

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None

    return np.concatenate([points,colors], axis=-1)


def load_jsonl(jsonl_path):
    """
    Loads a JSONL file and returns a list of dictionaries.
    Each line in the file should be a JSON object.
    """
    import json
    results = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results
    
        
    

pipe = DKTPipeline()



jsonl_data = load_jsonl(args.jsonl_path)
for line in tqdm(jsonl_data):


    cur_save_dir = join(args.save_dir, line['seq_name'])
    os.makedirs(cur_save_dir, exist_ok=True)

    mp4_file = line['rgb']
    

    videoreader = VideoReader(mp4_file)
    fps = videoreader.get_avg_fps()


    prediction = pipe(mp4_file,return_rgb=True)


    output_path = os.path.join(cur_save_dir, f'prediction.mp4')
    save_video(prediction['colored_depth_map'], output_path, fps=fps)



    total_frames = len(prediction['rgb_frames'])
    selected_frame_idx = list(range(total_frames))

    all_pc = pipe.prediction2pc(prediction['depth_map'], prediction['rgb_frames'], selected_frame_idx, return_pcd=True)

    # new_pcs = np.concatenate([x[None,...] for x in all_pc], axis=0)

    new_pcs = [pcd2pc_array(x) for x in all_pc]
    np.savez(f'{cur_save_dir}/pcd.npz', *new_pcs)


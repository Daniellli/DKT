from dkt.pipelines.pipelines import DKTPipeline
import os
from tools.common_utils import save_video


pipe = DKTPipeline()

demo_path = 'examples/1.mp4'
prediction = pipe(demo_path)


save_dir = 'logs'
os.makedirs(save_dir, exist_ok=True)
output_path = os.path.join(save_dir, 'demo.mp4')
save_video(prediction['colored_depth_map'], output_path, fps=25)



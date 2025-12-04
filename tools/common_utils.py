


import imageio
import numpy as np
from tqdm import tqdm


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):

    if len(frames) == 1 :
        frames[0].save(save_path.replace('.mp4', '.png'))
        return

    writer = imageio.get_writer(save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params)
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()
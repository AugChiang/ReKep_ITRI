import pyrealsense2 as rs
import cv2
import numpy as np
import os
from typing import Union, List, Tuple
from argparse import ArgumentParser

def cli():
    parser = ArgumentParser()
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="01.bag",
        help="Recording file name."
    )
    parser.add_argument(
        "-t", "--time",
        type=int,
        default=10, # 10s
        help="Time length of the video (sec)."
    )
    parser.add_argument(
        "-s", "--size",
        type=Union[tuple, list],
        default=[480,640],
        help="Image size (screen size) of (height, width)."
    )
    parser.add_argument(
        "-f", "--frame-rate",
        type=int,
        default=30,
        help="Recording frame rate."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save color and depth frames to output dir."
    )
    parser.add_argument(
        "--depth-dist",
        default=[1.,75.],
        type=Union[Tuple[float], List[float]],
        help="Numpy percentile to filter the depth data."
    )
    return parser.parse_args()

def start_recording(param):
    h, w = param.size
    fps = param.frame_rate
    output = param.output
    time_length = param.time

    os.makedirs("./recordings", exist_ok=True)
        
    pipeline = rs.pipeline()
    config = rs.config()

    align_to = rs.stream.color
    align = rs.align(align_to)

    # set output
    _, ext = os.path.splitext(output)
    path, _ = os.path.split(output)
    if len(ext) == 0 or ext != ".bag":        
        output = path + ".bag"
    # print("Output: ", output)
    # print("Path: ", path)
    frame_save_dir = os.path.join("./recordings" + path)
    # print("Frame save dir: ", frame_save_dir)
    config.enable_record_to_file(f"./recordings/{output}.bag")
    config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)

    pipeline.start(config)
    print("Recording...")

    if param.save:
        for i in range(time_length*fps):
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            save_frames(
                save_dir=frame_save_dir,
                file_name=f"frame_{str(i).rjust(3,'0')}",
                aligned_frames=aligned_frames,
                depth_dist=param.depth_dist
            )
    else:
        for i in range(time_length*fps):
            frames = pipeline.wait_for_frames()

    print("Done!")
    pipeline.stop()

def save_frames(
        save_dir:str,
        file_name:str,
        aligned_frames, 
        depth_dist:Union[List[float], Tuple[float]]=[1,75]
    ):
    """
    Save each RGB frame as png, D (depth) info as npy and D's colormap png (JET).

    Args:
        save_dir (str): "./recordings/<cli output path setting>"
        file_name (str): index of the frame.
        aligned_frames (_type_): RealSense aligned frames. see the `start_recording` func.
        depth_dist (Union[List[float], Tuple[float]], optional): percentile range to keep depth info. Defaults to [1,75].
    """
    # print("save dir: ", save_dir)
    img_save_dir = os.path.join(save_dir, "rgb")
    depth_save_dir = os.path.join(save_dir, "depth")
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(depth_save_dir, exist_ok=True)
    # print("img save dir: ", img_save_dir)
    # print("depth save dir: ", depth_save_dir)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    if not depth_frame or not color_frame:
        return
    depth_arr = np.asanyarray(depth_frame.get_data())
    color_arr = np.asanyarray(color_frame.get_data())
    valid_depth = depth_arr[depth_arr > 0]
    if valid_depth.size > 0:
        depth_min = np.percentile(valid_depth, min(depth_dist))
        depth_max = np.percentile(valid_depth, max(depth_dist))
        clipped_depth = np.clip(depth_arr, depth_min, depth_max)
        depth_normalized = cv2.convertScaleAbs(clipped_depth-depth_min, alpha=255.0 / (depth_max - depth_min))
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    else:
        depth_colormap = np.zeros_like(color_arr)  # fallback if no valid depth

    cv2.imwrite(f"{img_save_dir}/{file_name}.png", color_arr)
    cv2.imwrite(f"{depth_save_dir}/{file_name}.png", depth_colormap)
    np.save(f"{depth_save_dir}/{file_name}.npy", depth_arr)
    return



def main():
    param = cli()
    start_recording(param)
    
if __name__ == "__main__":
    main()
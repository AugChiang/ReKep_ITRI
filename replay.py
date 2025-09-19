import pyrealsense2 as rs
import cv2
import numpy as np
import os
from typing import Union, List, Tuple
from argparse import ArgumentParser
from glob import glob

def cli():
    parser = ArgumentParser()
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the recording file (.bag)"
    )
    return parser.parse_args()

def main(input:str, depth_dist:Union[List[float], Tuple[float]]=[1,75]):

    input_dir, _ = os.path.split(input) # path splits by "/" not "\\"
    input_list = glob(os.path.join(input_dir, "*.bag"))
    n = len(input_list)
    pipeline = rs.pipeline()
    config = rs.config()

    # read .bag file
    try:
        _, ext = os.path.splitext(input)
        if len(ext) == 0:
            input += ".bag"
        config.enable_device_from_file(input)
    except RuntimeError as e:
        print("Please check the file format (.bag only) or the path.")
        raise e
    curr_idx = input_list.index(input)

    # start play
    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    window_name = f"Playback - {input_list[curr_idx]}"
    print("Press 'ESC' to stop the replay.")
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

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

        images = np.hstack((color_arr, depth_colormap))
        cv2.imshow(window_name, images)

        key_pressed = cv2.waitKey(1) & 0xFF 
        if key_pressed == 27 : # ESC
            break
        
        # switch to another .bag file and replay
        
        if key_pressed == ord('a') and n > 1:
            curr_idx = (curr_idx-1)%n
            pipeline.stop()
            window_name = f"Playback - {input_list[curr_idx]}"
            cv2.destroyAllWindows()
            config.enable_device_from_file(input_list[curr_idx])
            pipeline.start(config)

        if key_pressed == ord('d') and n > 1:
            curr_idx = (curr_idx+1)%n
            pipeline.stop()
            window_name = f"Playback - {input_list[curr_idx]}"
            cv2.destroyAllWindows()
            config.enable_device_from_file(input_list[curr_idx])
            pipeline.start(config)

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    config = cli()
    main(config.input)
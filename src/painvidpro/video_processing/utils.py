"""Utility functions."""

import os
from contextlib import contextmanager
from typing import Dict

import cv2


# Custom context manager for cv2.VideoCapture
@contextmanager
def video_capture_context(video_path: str):
    """Context Manager to manage cv2.VideoCapture.

    Args:
        video_path: The path to the video.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        yield cap
    finally:
        cap.release()


def extract_frames(video_path: str, output_folder: str, target_fps: int = 4) -> Dict[int, str]:
    """Extracts the frames of a video.

    Args:
        video_path: The path of the video.
        output_folder: The output folder to store the frames.
        target_fps: Number of frames to extract per second.

    Returns:
        A dict storing the frame index as key and the name of the frame on the disk as the value.
    """
    ret_dict: Dict[int, str] = {}
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with video_capture_context(video_path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / target_fps)
        frame_count: int = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_name = f"frame_{frame_count:08d}.jpg"
                frame_filename = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_filename, frame)
                ret_dict[frame_count] = frame_name
            frame_count += 1
    return ret

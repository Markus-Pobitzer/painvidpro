"""Utility functions."""

import os
from contextlib import contextmanager
from typing import Dict, List

import cv2
import ffmpeg
import numpy as np


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


# Custom context manager for cv2.VideoCapture
@contextmanager
def video_writer_context(output_path: str, width: int, height: int, fourcc: str = "mp4v", fps: int = 30):
    """Context Manager to manage cv2.VideoWriter.

    Args:
        video_path: The path to the video.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    try:
        yield video_writer
    finally:
        video_writer.release()


def save_frames_as_video(frame_list: List[np.ndarray], output_path: str, fps: int = 30):
    """Save the frames in the frame_list as a video.

    Args:
        frame_list: The list of frames.
        output_path: The path to save the video.
        fps: The frames per second.
    """
    height, width, _ = frame_list[0].shape
    with video_writer_context(output_path, width=width, height=height, fps=fps) as video_writer:
        for frame in frame_list:
            video_writer.write(frame)


def extract_frames(
    video_path: str, output_folder: str, target_fps: int = -1, file_ending: str = "png"
) -> Dict[int, str]:
    """Extracts the frames of a video.

    Args:
        video_path: The path of the video.
        output_folder: The output folder to store the frames.
        target_fps: Number of frames to extract per second. If
            it is smaller equals 0, save all frames.
        file_ending: The file ending.

    Returns:
        A dict storing the frame index as key and the name of the frame on the disk as the value.
    """
    ret_dict: Dict[int, str] = {}
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with video_capture_context(video_path) as cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / target_fps) if target_fps > 0 else 0
        frame_count: int = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_interval == 0 or (frame_count % frame_interval == 0):
                frame_name = f"frame_{frame_count:08d}.{file_ending}"
                frame_filename = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_filename, frame)
                ret_dict[frame_count] = frame_name
            frame_count += 1
    return ret


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.5) -> np.ndarray:
    """
    Overlays the mask on top of the image with a specified color and transparency.

    Args:
        image: The original image as a numpy array in cv2 BGR format.
        mask: The mask as a numpy array.
        color: The color to use for the overlay (default is blue).
        alpha: The transparency level of the overlay (default is 0.5).

    Returns:
        A numpy array with the overlay.
    """
    # Create a color overlay image
    overlay = np.zeros_like(image, dtype=np.uint8)
    overlay[mask > 0] = color

    # Blend the original image with the overlay using the specified transparency
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return blended


def median_of_video(video_path: str, num_samples: int = 100) -> np.ndarray:
    """Computes the median of the video.

    Samples num_samples frames from the video, get median
    based on the samples.

    Args:
        video_path: The path to the video.
        num_samples: The number of samples.

    Returns:
        The median image.
    """
    # Using ffmpeg since cv2 may not be reliable see:
    # https://github.com/opencv/opencv/issues/9053

    # Probe the video to get its properties
    probe = ffmpeg.probe(video_path)
    video_info = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    width = int(video_info["width"])
    height = int(video_info["height"])
    length = int(video_info["nb_frames"])

    # In case there are more samples than frames
    num_samples = min(num_samples, length)
    # sampled_frames = np.zeros((num_samples, height, width, 3), dtype=np.uint8)

    # Generate frame indices to sample
    frame_indices = np.linspace(0, length - 1, num=num_samples, dtype=int)

    select_filter = "+".join(f"eq(n,{idx})" for idx in frame_indices)

    # Use ffmpeg to extract the specific frame
    out, _ = (
        ffmpeg.input(video_path)
        .filter("select", select_filter)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", vsync="0")
        .run(capture_stdout=True, capture_stderr=True)
    )
    sampled_frames = np.frombuffer(out, dtype=np.uint8).reshape(-1, height, width, 3)
    return np.median(sampled_frames, axis=0)

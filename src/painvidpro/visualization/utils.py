"""Utility for visualization."""

import json
import os
import tempfile
from os import listdir
from os.path import isfile, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from painvidpro.utils.metadata import load_metadata
from painvidpro.video_processing.utils import video_writer_context


def get_video_folders(root_folder: str) -> List[str]:
    """Gets all video folders from a root directory.

    Assuming following folder structure:
    root/
       source_1/
          video_folder_1/
          ...
       ...

    Args:
        root_folder: The path to the root folder.

    Returns:
        A list of paths to the video folders.
    """
    video_folders: List[str] = []
    for source in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, source)
        if os.path.isdir(subfolder_path):
            for sub_subfolder in os.listdir(subfolder_path):
                sub_subfolder_path = os.path.join(subfolder_path, sub_subfolder)
                if os.path.isdir(sub_subfolder_path):
                    video_folders.append(sub_subfolder_path)
    return video_folders


def get_keyframe_metadata(metadata: Dict[str, Any]) -> Tuple[Tuple[int, int], Optional[List[List[int]]], List[int]]:
    """Extaracts start/end frame idx, keyframe list and selected keyframes.

    Args:
        metadata: The metadata dict.

    Returns:
        Start/end frame idx, keyframe list and selected keyframes.
    """
    start_frame = metadata.get("start_frame_idx", -1)
    end_frame = metadata.get("end_frame_idx", -1)

    keyframe_list = metadata.get("keyframe_list", None)
    selected_keyframe_list = metadata.get("selected_keyframe_list", [])
    return (start_frame, end_frame), keyframe_list, selected_keyframe_list


def load_pipeline(root_folder: str):
    """Loads the pipeline .json."""
    with open(os.path.join(root_folder, "pipeline.json"), "r") as f:
        pipeline = json.load(f)
    return pipeline


def filter_processed_metadata(sub_subfolders: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Function to filter metadata based on entries."""
    processed_metadata = []
    for sub_subfolder in sub_subfolders:
        succ, metadata = load_metadata(Path(sub_subfolder))
        if not succ:
            continue
        (start_frame, end_frame), keyframe_list, selected_keyframe_list = get_keyframe_metadata(metadata)
        # Only take samples that have been successfully processed with the Keyframe Processor
        if start_frame < 0 or end_frame < 0 or keyframe_list is None or len(selected_keyframe_list) < 2:
            continue
        processed_metadata.append((sub_subfolder, metadata))
    return processed_metadata


def filter_processed_metadata_extracted_frames(sub_subfolders: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Function to filter metadata based on entries."""
    processed_metadata = []
    for sub_subfolder in sub_subfolders:
        succ, metadata = load_metadata(Path(sub_subfolder))
        if not succ:
            continue

        start_frame = metadata.get("start_frame_idx", -1)
        end_frame = metadata.get("end_frame_idx", -1)
        extracted_frames = metadata.get("extracted_frames", [])
        # Only take samples that have been successfully processed with the Keyframe Processor
        if start_frame < 0 or end_frame < 0 or len(extracted_frames) == 0:
            continue
        processed_metadata.append((sub_subfolder, metadata))
    return processed_metadata


def get_video_path(sub_subfolder, video_name: str = "video.mp4"):
    """Returns the video path."""
    return os.path.join(sub_subfolder, video_name)


def load_video_and_keyframes(
    sub_subfolder: str, metadata: Dict[str, Any], video_name: str = "video.mp4"
) -> Tuple[str, List[int]]:
    """Function to load video and keyframes"""
    video_path = os.path.join(sub_subfolder, video_name)
    keyframes = metadata.get("selected_keyframe_list", [])
    return video_path, keyframes


def display_keyframes(video_path: str, keyframes: List[int]) -> List[np.ndarray]:
    """Function to display keyframes in RGB space."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for frame_idx in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


def get_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Loads the frame from the video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Was not able to laod frame with index {frame_idx} from {video_path}")
    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def compute_progress_dist(metadata: Dict[str, Any], number_bins: int = 100) -> List[int]:
    """Each Video from start frame to last keaframe gets ordered in to bins."""
    start_frame_idx = metadata["start_frame_idx"]
    last_keyframe_idx = metadata["selected_keyframe_list"][-1] - start_frame_idx
    progress_bin_list = [0] * number_bins
    for sele_keyframe in metadata["selected_keyframe_list"][:-1]:
        progress = (sele_keyframe - start_frame_idx) / last_keyframe_idx
        prog_bin = int(progress * number_bins)
        progress_bin_list[prog_bin] = 1
    return progress_bin_list


def vis_progress_distribution(progress_dist: List[int], width=1000, height=25) -> np.ndarray:
    """Creates a visual representation of the progress distribution."""
    n = len(progress_dist)
    bin_width = (width - (n - 1) * 2) // n  # Calculate the width of each bin
    image = np.zeros((height, width, 3), dtype=np.uint8)

    current_x = 0
    for value in progress_dist:
        color = (0, 255, 0) if value == 1 else (255, 127, 127)  # Green for 1, Red for 0
        image[:, current_x : current_x + bin_width] = color
        current_x += bin_width
        if current_x < width:
            image[:, current_x : current_x + 2] = (255, 255, 255)  # White vertical line
            current_x += 2

    return image


def get_reference_frame(reference_frame_path: str, video_path: str, keyframes: List[int]) -> np.ndarray:
    """Loads the reference frame."""
    img = cv2.imread(reference_frame_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    try:
        return get_frame(video_path, keyframes[-1])
    except Exception as _:
        return np.zeros((250, 250, 3)) + (255, 0, 0)


def create_temp_file(suffix: str = ".mp4") -> str:
    """Function to create a temporary file and return its path."""
    # Create a temporary file with a .mp4 extension
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_file_path = temp_file.name
    temp_file.close()  # Close the file but keep it on disk
    return temp_file_path


def cleanup(temp_file_path: str):
    """Cleanup function to delete the temporary file."""
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


def save_video_from_frames(
    video_dir: str, frame_path_list: List[Any], video_output_path: str, fps: int = 1
) -> Optional[str]:
    """Saves the frames from frame_path_list as a video to video_output_path"""
    frame_list = [join(video_dir, f["path"]) for f in frame_path_list if isfile(join(video_dir, f["path"]))]

    if len(frame_list) == 0:
        return None

    img = cv2.imread(frame_list[0])
    height, width, _channels = img.shape
    with video_writer_context(video_output_path, width, height, fps=fps) as vid_out:
        for img_path in frame_list:
            frame = cv2.imread(img_path)
            vid_out.write(frame)
    return video_output_path


def load_log_files(folder: str, file_ending: str = ".log"):
    """Loads the name and log file into a list."""
    ret = []
    for f in listdir(folder):
        f_path = join(folder, f)
        if isfile(f_path) and f.endswith(file_ending):
            ret.append((f[: -len(file_ending)], f_path))
    return ret


def read_file(file_path):
    """Function to read the content of a log file."""
    try:
        with open(file_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        return "Log file not found."

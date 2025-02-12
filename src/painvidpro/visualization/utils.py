"""Utility for visualization."""

import os
from typing import Any, Dict, List, Optional, Tuple


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

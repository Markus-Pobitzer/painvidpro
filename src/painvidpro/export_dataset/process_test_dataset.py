"""Processes and saves the test dataset."""

from typing import Any, Dict

import numpy as np
from datasets import Dataset
from PIL import Image

from painvidpro.utils.image_processing import resize_and_center_crop


def resize_map(entry: Dict[str, Any], width: int = 512, height: int = 512) -> Dict[str, Any]:
    """Resize and center crop map function."""
    return {
        "reference_frame": entry["reference_frame"].resize((width, height)),
        "frame": entry["frame"].resize((width, height)),
    }


def resize_and_center_crop_map(entry: Dict[str, Any], width: int = 512, height: int = 512) -> Dict[str, Any]:
    """Resize and center crop map function."""
    ref_frame = np.array(entry["reference_frame"])
    frame = np.array(entry["frame"])
    return {
        "reference_frame": Image.fromarray(
            resize_and_center_crop(ref_frame, target_width=width, target_height=height)
        ),
        "frame": Image.fromarray(resize_and_center_crop(frame, target_width=width, target_height=height)),
    }


def apply_img_processing(ds: Dataset, config: Dict[str, Any]) -> Dataset:
    """Applies different mapping fucntions on the dataset.

    Args:
        ds: The Dataset.
        config: The settings used.

    Returns:
        A Dataset.
    """
    width = config.get("width", 512)
    height = config.get("height", 512)
    ret_dataset: Dataset
    if config["map_function"] == "resize_and_center_crop":
        ret_dataset = ds.map(function=resize_and_center_crop_map, fn_kwargs={"width": width, "height": height})
    elif config["map_function"] == "resize":
        ret_dataset = ds.map(function=resize_map, fn_kwargs={"width": width, "height": height})
    else:
        raise ValueError(f"Map function {config['map_function']} not supported.")
    return ret_dataset

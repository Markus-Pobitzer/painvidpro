"""Utility for processing images."""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def process_input(obj: Union[np.ndarray, str, cv2.VideoCapture], convert_bgr_to_rgb: bool = False) -> np.ndarray:
    """
    Processes the input object and returns an image in np.ndarray format.

    Args:
        obj: Input object which can be either an np.ndarray, a string representing
            an image path, or a cv2.VideoCapture object.
        convert_bgr_to_rgb: If set swaps the bgr channels to get rgb channels.

    Returns:
        np.ndarray: The image in np.ndarray format.

    Raises:
        TypeError: If the input is neither an np.ndarray, a string, nor a cv2.VideoCapture object.
        ValueError: If the string does not correspond to a valid image path or if the video capture fails.
    """
    ret: np.ndarray
    if isinstance(obj, np.ndarray):
        ret = obj
    elif isinstance(obj, str):
        try:
            image = cv2.imread(obj)
            if image is None:
                raise ValueError("The provided string does not correspond to a valid image path.")
            ret = image
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")
    elif isinstance(obj, cv2.VideoCapture):
        ret, frame = obj.read()
        if not ret:
            raise ValueError("Failed to read frame from video capture.")
        ret = frame
    else:
        raise TypeError(
            "Input must be either an np.ndarray, a str representing an image path, or a cv2.VideoCapture object."
        )

    if convert_bgr_to_rgb:
        ret = cv2.cvtColor(ret, cv2.COLOR_BGR2RGB)

    return ret


def convert_cv2_to_pil(img: np.ndarray) -> Image:
    """Converts an image stored in cv2 fromat to a PIL Image.

    Args:
        img: Image in cv2 format, assuming BGR channel ordering.

    Returns:
        An image in PIL format.
    """
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def find_best_aspect_ratio(image_size: Tuple[int, int], size_list: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """
    Find the best image size in size_list that preserves the
    original image's aspect ratio the closests.

    Args:
        image_size: Original image dimensions (width, height).
        size_list: List of candidate sizes [(width, height), ...].

    Returns:
        tuple: Best matching model size, or None if size_list is empty.
    """
    original_width, original_height = image_size
    original_ratio = original_width / original_height
    best_size = None
    min_diff = float("inf")

    for size in size_list:
        model_width, model_height = size
        model_ratio = model_width / model_height
        ratio_diff = abs(original_ratio - model_ratio)

        # Update best size if current model is better match
        if ratio_diff < min_diff:
            min_diff = ratio_diff
            best_size = size
        elif ratio_diff == min_diff:
            pass

    return best_size


def resize_and_center_crop(
    image: np.ndarray, target_width: int, target_height: int, interpolation=cv2.INTER_AREA
) -> np.ndarray:
    """Resize and center crop an image.

    Keeps aspect ratio.
    Code from https://github.com/lllyasviel/Paints-UNDO.

    Args:
        image: The image as np.ndarray.
        target_width: The desired width.
        target_height: The desired height.
        interpolation: The interporlation setting.

    Return:
        The resized and center cropped image as an array.
    """
    original_height, original_width = image.shape[:2]
    k = max(target_height / original_height, target_width / original_width)
    new_width = int(round(original_width * k))
    new_height = int(round(original_height * k))
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    x_start = (new_width - target_width) // 2
    y_start = (new_height - target_height) // 2
    cropped_image = resized_image[y_start : y_start + target_height, x_start : x_start + target_width]
    return cropped_image

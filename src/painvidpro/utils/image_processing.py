"""Utility for processing images."""

from typing import Union

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

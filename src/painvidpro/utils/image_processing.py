"""Utility for processing images."""

from typing import Union

import cv2
import numpy as np


def process_input(obj: Union[np.ndarray, str, cv2.VideoCapture]) -> np.ndarray:
    """
    Processes the input object and returns an image in np.ndarray format.

    Args:
        obj: Input object which can be either an np.ndarray, a string representing an image path, or a cv2.VideoCapture object.

    Returns:
        np.ndarray: The image in np.ndarray format.

    Raises:
        TypeError: If the input is neither an np.ndarray, a string, nor a cv2.VideoCapture object.
        ValueError: If the string does not correspond to a valid image path or if the video capture fails.
    """
    if isinstance(obj, np.ndarray):
        return obj
    elif isinstance(obj, str):
        try:
            image = cv2.imread(obj)
            if image is None:
                raise ValueError("The provided string does not correspond to a valid image path.")
            return image
        except Exception as e:
            raise ValueError(f"An error occurred while loading the image: {e}")
    elif isinstance(obj, cv2.VideoCapture):
        ret, frame = obj.read()
        if not ret:
            raise ValueError("Failed to read frame from video capture.")
        return frame
    else:
        raise TypeError(
            "Input must be either an np.ndarray, a str representing an image path, or a cv2.VideoCapture object."
        )

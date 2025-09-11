"""Utility for processing images."""

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps


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
    original image's aspect ratio the closest.

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


def pil_resize_and_center_crop(image: Image.Image, resolution: int):
    """Resize and center crop.

    Resizes smaller side to resolution while keeping aspect ratio.
    Center crop to (resolution, resolution).

    Args:
        image: PIL Image.
        resolution: The final resolution.

    Returns:
        The resized and center cropped PIL Image.
    """
    width, height = image.size
    if width < height:
        new_width = resolution
        new_height = int(resolution * height / width)
    else:
        new_height = resolution
        new_width = int(resolution * width / height)

    image = image.resize((new_width, new_height), Image.BILINEAR)

    # Step 2: Center crop to (resolution, resolution)
    left = (new_width - resolution) // 2
    top = (new_height - resolution) // 2
    right = left + resolution
    bottom = top + resolution

    image = image.crop((left, top, right, bottom))
    return image


def pil_resize(
    image: Image.Image,
    target_size: Tuple[int, int],
    pad_input: bool = False,
    padding_color: Union[str, int, Tuple[int, ...]] = "white",
) -> Image.Image:
    """Resizing it to the target size.

    Args:
        image: Input image to be processed.
        target_size: Target size (width, height).
        pad_input: If set resizes the image while keeping the aspect ratio and pads the unfilled part.
        padding_color: The color for the padded pixels.

    Returns:
        The resized image
    """
    if pad_input:
        # Resize image, keep aspect ratio
        image = ImageOps.contain(image, size=target_size)
        # Pad left side and bottom
        image = ImageOps.pad(image, size=target_size, color=padding_color)
    else:
        image = image.resize(target_size)
    return image


def pil_resize_with_padding(img: Image.Image, target_size: Tuple[int, int] = (1024, 1024), padding_color: int = 0):
    """
    Resize the image to the target size while maintaining the aspect ratio, and pad the left or top with the specified color (default black) if necessary.

    You can also call pil_resize(image=img, target_size=target_size, pad_input=True).

    The image is resized such that:
        - If the original image is larger than the target in either dimension, it is scaled down to fit within the target.
        - If the original image is smaller in both dimensions, it is scaled up to fill the target as much as possible without exceeding,
          choosing the scaling factor based on which dimension (height or width) is closer to the target.

    After resizing, the image is placed at the bottom-right of the target canvas, so that the left and/or top areas are padded.

    Args:
        img (PIL.Image): The input image.
        target_size (tuple): The target size in pixels as (width, height).
        padding_color (int or tuple, optional): The color to use for padding. Default is 0 (black). For multi-band images, use a tuple.

    Returns:
        PIL.Image: The resized and padded image.
    """
    orig_w, orig_h = img.size
    target_w, target_h = target_size

    if orig_w == 0 or orig_h == 0:
        return Image.new(img.mode, (target_w, target_h), padding_color)

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized_img = img.resize((new_w, new_h), Image.LANCZOS)

    new_img = Image.new(img.mode, (target_w, target_h), padding_color)
    x = target_w - new_w
    y = target_h - new_h
    new_img.paste(resized_img, (x, y))

    return new_img


def pil_reverse_resize_with_padding(padded_img: Image.Image, original_resolution: Tuple[int, int]):
    """
    Reverse the `resize_with_padding` operation to retrieve the resized image (without padding).

    This function crops the non-padded region (the resized image) from the padded image.
    The non-padded region is determined by the original resolution and target size using
    the same scaling logic as the forward operation.

    Args:
        padded_img (PIL.Image): Padded image created by resize_with_padding.
        original_resolution (tuple): Original image dimensions (width, height).

    Returns:
        PIL.Image: The resized image without padding.

    Raises:
        ValueError: If original_resolution contains non-positive dimensions.
    """
    orig_w, orig_h = original_resolution
    target_w, target_h = padded_img.size

    # Validate dimensions
    if orig_w <= 0 or orig_h <= 0:
        raise ValueError("Original dimensions must be positive integers")

    # Calculate scaling factor used in forward operation
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # Calculate position of resized image (bottom-right placement)
    x = target_w - new_w
    y = target_h - new_h

    # Crop and return the resized image
    ret = padded_img.crop((x, y, x + new_w, y + new_h))
    ret = ret.resize((orig_w, orig_h), Image.LANCZOS)
    return ret

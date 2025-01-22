"""Utility functions for image differences."""

import cv2
import numpy as np


def image_diff_mask(
    frame: np.ndarray,
    prev_frame: np.ndarray,
    is_greater_than_threshold: bool,
    keyframe_diff_threshold: int = 8,
    blur_frames: bool = False,
    blur_kernel_size: int = 3,
    open_mask: bool = False,
    open_kernel: int = 5,
) -> np.ndarray:
    """
    Extracts a mask based on the difference of the current frame and the previous frame.

    Args:
        frame: The current frame.
        prev_frame: the previous frame.
        is_greater_than_threshold: I set to
            True: Mask entry is true if difference > keyframe_diff_threshold
            False: Mask entry is true if difference <= keyframe_diff_threshold
        keyframe_diff_threshold: The threshold to set the difference.
        blur_frames: If set blurs the frames before computing the difference.
        blur_kernel_size: Kernel size of the Gaussian Blur.
        open_mask: If set erodes the mask before. This helps removing
            minor changes not associated to the occlusion. Afterwards
            a dilation gets applied to restore the size of the occlusion.
        open_kernel: Kernel size of the opening operation.

    Returnrs:
        A mask indicating where pixel changes were noticable.
    """
    kernel = np.ones((open_kernel, open_kernel), np.uint8)
    if blur_frames:
        frame = cv2.GaussianBlur(frame, (blur_kernel_size, blur_kernel_size), 1.0)
        prev_frame = cv2.GaussianBlur(prev_frame, (blur_kernel_size, blur_kernel_size), 1.0)

    first_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    prev_lab = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2Lab)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    d = np.sqrt(np.sum((first_lab - prev_lab) ** 2, axis=-1))
    if is_greater_than_threshold:
        mask[d > keyframe_diff_threshold] = 255
    else:
        mask[d <= keyframe_diff_threshold] = 255

    if open_mask:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

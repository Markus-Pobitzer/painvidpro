"""Utility functions for image differences."""

import cv2
import numpy as np


def image_diff(
    frame: np.ndarray,
    other_frame: np.ndarray,
    blur_frames: bool = False,
    blur_kernel_size: int = 3,
) -> np.ndarray:
    """
    Extracts the difference of the frame and the other frame.

    Args:
        frame: The current frame.
        other_frame: the other frame.
        blur_frames: If set blurs the frames before computing the difference.
        blur_kernel_size: Kernel size of the Gaussian Blur.

    Returnrs:
        An np.ndarray indicating the L2 pixel difference of the two frames.
    """
    if blur_frames:
        frame = cv2.GaussianBlur(frame, (blur_kernel_size, blur_kernel_size), 1.0)
        other_frame = cv2.GaussianBlur(other_frame, (blur_kernel_size, blur_kernel_size), 1.0)

    first_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
    prev_lab = cv2.cvtColor(other_frame, cv2.COLOR_BGR2Lab)

    return np.sqrt(np.sum((first_lab - prev_lab) ** 2, axis=-1))


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
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    d = image_diff(frame, prev_frame, blur_frames=blur_frames, blur_kernel_size=blur_kernel_size)
    if is_greater_than_threshold:
        mask[d > keyframe_diff_threshold] = 255
    else:
        mask[d <= keyframe_diff_threshold] = 255

    if open_mask:
        kernel = np.ones((open_kernel, open_kernel), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

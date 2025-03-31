"""Implementation of OcclusionMaskingNoChnges."""

from typing import List, Union

import cv2
import numpy as np

from painvidpro.logo_masking.base import LogoMaskingBase
from painvidpro.utils.image_diff import image_diff
from painvidpro.utils.image_processing import process_input


class LogoMaskingNoChanges(LogoMaskingBase):
    def __init__(self):
        """Class to compute logo masks."""
        super().__init__()
        self.set_default_parameters()

    def set_default_parameters(self):
        self.params = {
            "keyframe_diff_threshold": 8,
            "blur_frames": False,
            "blur_kernel_size": 3,
        }

    def check_no_changes(
        self,
        frame_list: Union[List[np.ndarray], List[str], cv2.VideoCapture],
        num_frames: int,
        keyframe_diff_threshold: int = 8,
        blur_frames: bool = False,
        blur_kernel_size: int = 3,
    ) -> np.ndarray:
        """
        Accumulates the entries of the masks into one final mask.

        Args:
            frame_list: The List of frames.
            num_frames: The number of frames to check, shuld be length of frame_lists.
            keyframe_diff_threshold: The threshold to set the difference.
            blur_frames: If set blurs the frames before computing the difference.
            blur_kernel_size: Kernel size of the Gaussian Blur.

        Returnrs:
            A mask indicating where no pixel movements were noticable.

        Raises:
            ValueError if frames in frame_list can not be proessed.
        """
        if isinstance(frame_list, cv2.VideoCapture):
            prev_frame = process_input(frame_list)
        else:
            prev_frame = process_input(frame_list[0])

        mask = np.zeros(prev_frame.shape[:2], dtype=np.uint8) + 255
        for i in range(1, num_frames):
            if isinstance(frame_list, cv2.VideoCapture):
                frame = process_input(frame_list)
            else:
                frame = process_input(frame_list[i])

            diff = image_diff(frame, prev_frame, blur_frames=blur_frames, blur_kernel_size=blur_kernel_size)
            mask[diff > keyframe_diff_threshold] = 0

        return mask

    def compute_mask_list(self, frame_list: List[np.ndarray], offload_model: bool = True) -> List[np.ndarray]:
        """Logo mask indicating no changes.

        All frames get analyzed and checked which pixels do not
        change throughout the whole sequence.

        Args:
            frame_list: List of frames in cv2 image format.
            offload_model: Gets ignored.

        Returns:
            List with one mask.
        """
        keyframe_diff_threshold = self.params.get("keyframe_diff_threshold", 8)
        blur_frames = self.params.get("blur_frames", False)
        blur_kernel_size = self.params.get("blur_kernel_size", 3)

        num_frames = len(frame_list)
        mask = self.check_no_changes(
            frame_list,
            num_frames,
            keyframe_diff_threshold=keyframe_diff_threshold,
            blur_frames=blur_frames,
            blur_kernel_size=blur_kernel_size,
        )
        return [mask] * num_frames

"""Implementation of OcclusionMaskingAccumulateDiff."""

from typing import List

import cv2
import numpy as np

from painvidpro.occlusion_masking.base import OcclusionMaskingBase


class OcclusionMaskingAccumulateDiff(OcclusionMaskingBase):
    def __init__(self):
        """Class to compute occlusion masks."""
        super().__init__()
        self.set_default_parameters()

    def set_default_parameters(self):
        self.params = {
            "past_num_frames": 30,
            "future_num_frames": 2,
            "keyframe_diff_threshold": 8,
            "erode_dilate_mask": False,
            "erosion_dilation_kernel_size": 3,
            "blur_frames": False,
            "blur_kernel_size": 3,
        }

    def image_diff_mask(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        keyframe_diff_threshold: int = 8,
        erode_dilate_mask: bool = False,
        erosion_dilation_kernel_size: int = 3,
        blur_frames: bool = False,
        blur_kernel_size: int = 3,
    ) -> np.ndarray:
        """
        Extracts a mask based on the difference of the current frame and the previous frame.

        Args:
            frame: The current frame.
            prev_frame: the previous frame.
            keyframe_diff_threshold: The threshold to set the difference.
            erode_dilate_mask: If set erodes the mask before. This helps removing
                minor movements not associated to the occlusion. Afterwards
                a dilation gets applied to restore the size of the occlusion.
            erosion_dilation_kernel_size: Kernel size of erosion and dilation.
            blur_frames: If set blurs the frames before computing the difference.
            blur_kernel_size: Kernel size of the Gaussian Blur.

        Returnrs:
            A mask indicating where pixel movements were noticable.
            If check_if_queue_is_full is set and last_frame_queue is not
            full the mask is set everywhere.
        """
        kernel = np.ones((erosion_dilation_kernel_size, erosion_dilation_kernel_size), np.uint8)
        blur_kernel = np.ones((blur_kernel_size, blur_kernel_size), np.uint8)
        if blur_frames:
            frame = cv2.GaussianBlur(frame, blur_kernel, 1.0)
            prev_frame = cv2.GaussianBlur(prev_frame, blur_kernel, 1.0)

        first_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        prev_lab = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2Lab)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        d = np.sqrt(np.sum((first_lab - prev_lab) ** 2, axis=-1))
        mask[d > keyframe_diff_threshold] = 255

        if erode_dilate_mask:
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=3)

        return mask

    def accumulate_movement_mask(
        self,
        mask_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        Accumulates the entries of the masks into one final mask.

        Args:
            mask_list: The List of masks.

        Returnrs:
            A mask indicating where pixel movements were noticable.

        Raises:
            ValueError if mask_lsit is empty.
        """
        if len(mask_list) == 0:
            raise ValueError("mask_list is empty.")

        mask = mask_list[0]
        for next_mask in mask_list[1:]:
            mask = np.maximum(mask, next_mask)

        return mask

    def compute_mask_list(self, frame_list: List[np.ndarray], offload_model: bool = True) -> List[np.ndarray]:
        """Occlusion masks based on the image difference.

        The image difference gets computed on the past_num_frames
        and  future_num_frames.

        Args:
            frame_list: List of frames in cv2 image format.
            offload_model: Gets ignored.

        Returns:
            List of masks, one for each input frame where an occlusion may exist.
        """
        mask_list: List[np.ndarray] = []
        past_num_frames = self.params.get("past_num_frames", 30)
        future_num_frames = self.params.get("future_num_frames", 2)
        keyframe_diff_threshold = self.params.get("keyframe_diff_threshold", 8)
        erode_dilate_mask = self.params.get("erode_dilate_mask", False)
        erosion_dilation_kernel_size = self.params.get("erosion_dilation_kernel_size", 3)
        blur_frames = self.params.get("blur_frames", False)
        blur_kernel_size = self.params.get("blur_kernel_size", 3)

        # Compute the diff of current frame and previous one
        diff_mask_list: List[np.ndarray] = []
        for i, frame in enumerate(frame_list):
            if i == 0:
                diff_mask_list.append(np.zeros(frame.shape[:2], dtype=np.float64))
            else:
                diff_mask_list.append(
                    self.image_diff_mask(
                        frame,
                        frame_list[i - 1],
                        keyframe_diff_threshold=keyframe_diff_threshold,
                        erode_dilate_mask=erode_dilate_mask,
                        erosion_dilation_kernel_size=erosion_dilation_kernel_size,
                        blur_frames=blur_frames,
                        blur_kernel_size=blur_kernel_size,
                    )
                )

        # Accumulate the differences to a mask
        for i in range(len(diff_mask_list)):
            mask_list.append(
                self.accumulate_movement_mask(
                    diff_mask_list[max(0, i - past_num_frames) : min(len(diff_mask_list), i + future_num_frames)]
                )
            )
        return mask_list

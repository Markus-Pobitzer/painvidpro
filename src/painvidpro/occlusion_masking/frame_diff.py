"""Implementation of OcclusionMaskingFrameDiff."""

from typing import List

import cv2
import numpy as np

from painvidpro.occlusion_masking.base import OcclusionMaskingBase


class OcclusionMaskingFrameDiff(OcclusionMaskingBase):
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

    def compute_movement_mask(
        self,
        frame: np.ndarray,
        frame_list: List[np.ndarray],
        keyframe_diff_threshold: int = 8,
        erode_dilate_mask: bool = False,
        erosion_dilation_kernel_size: int = 3,
        blur_frames: bool = False,
        blur_kernel_size: int = 3,
    ) -> np.ndarray:
        """
        Extracts a mask based on the difference of the current frame and the last n frames.

        Args:
            frame: The current frame.
            frame_list: List of the frames to compute the difference.
            erode_dilate_mask: If set erodes the past movements mask before
                appliying it to the final mask. This helps removing
                minor movements not associated to the occlusion. Afterwards
                a dilation gets applied to restore the size of the occlusion.
            erosion_dilation_kernel_size: Kernel size of erosion and dilation.
            blur_frames: If set blurs the frames before computing the difference.
            blur_kernel_size: Kernel size of the Gaussian Blur.

        Returnrs:
            A mask indicating where pixel movements were noticable.
        """
        kernel = np.ones((erosion_dilation_kernel_size, erosion_dilation_kernel_size), np.uint8)
        blur_kernel = np.ones((blur_kernel_size, blur_kernel_size), np.uint8)
        if blur_frames:
            frame = cv2.GaussianBlur(frame, blur_kernel, 1.0)
        first_gaussian = frame
        first_lab = cv2.cvtColor(first_gaussian, cv2.COLOR_BGR2Lab)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for past_frame in frame_list:
            if blur_frames:
                past_frame = cv2.GaussianBlur(past_frame, (3, 3), 1.0)
            past_lab = cv2.cvtColor(past_frame, cv2.COLOR_BGR2Lab)

            d = np.sqrt(np.sum((first_lab - past_lab) ** 2, axis=-1))
            mask[d > keyframe_diff_threshold] = 255

            if erode_dilate_mask:
                mask = cv2.erode(mask, kernel, iterations=1)
                mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def compute_mask_list(self, frame_list: List[np.ndarray]) -> List[np.ndarray]:
        """Occlusion masks based on the image difference.

        The image difference gets computed on the past_num_frames
        and  future_num_frames.

        Args:
            frame_list: List of frames in cv2 image format.

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

        for i in range(len(frame_list)):
            mask_list.append(
                self.compute_movement_mask(
                    frame_list[i],
                    frame_list[max(0, i - past_num_frames) : i]
                    + frame_list[(i + 1) : min(len(frame_list), i + future_num_frames)],
                    keyframe_diff_threshold=keyframe_diff_threshold,
                    erode_dilate_mask=erode_dilate_mask,
                    erosion_dilation_kernel_size=erosion_dilation_kernel_size,
                    blur_frames=blur_frames,
                    blur_kernel_size=blur_kernel_size,
                )
            )
        return mask_list

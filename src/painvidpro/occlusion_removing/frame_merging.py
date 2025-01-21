"""Merges frames to remove occlusions."""

from typing import List, Tuple

import cv2
import numpy as np

from painvidpro.occlusion_removing.base import OcclusionRemovingBase


class OcclusionRemovingFrameMerging(OcclusionRemovingBase):
    def __init__(self):
        """Class to remove occlusions."""
        super().__init__()
        self.set_default_parameters()

    def set_default_parameters(self):
        self.params = {
            "fill_color": (255, 255, 255),
            "leading_zeros": 8,
            "delete_intermediate_data": True,
        }

    def create_forward_backward_frames_and_masks_on_disk(
        self,
        frame_path_list: List[str],
        mask_path_list: List[str],
        output_dir: str,
        fill_color: Tuple[int, int, int],
        leading_zeros: int = 8,
    ):
        """
        Creates forward and backward frames and masks for a sequence of frames and masks, storing them on disk.

        Args:
            frame_path_list: List of paths to frames representing the video.
            mask_path_list: List of paths to masks corresponding to each frame.
                The mask is either 0 (the pixel is good to use) or 255 (the pixel may be occluded).
            fill_color: The color that gets filled in the occluded region of the
                first and last frame if no other viable pixel was found.
            output_dir: Directory where the output frames and masks will be stored.
            leading_zeros: Number of leading zeros before frame index.

        Returns:
            forward_frame_paths: List of paths to forward frames.
            forward_mask_paths: List of paths to forward masks.
            backward_frame_paths: List of paths to backward frames.
            backward_mask_paths: List of paths to backward masks.
        """
        num_frames = len(frame_path_list)
        height, width, _ = cv2.imread(frame_path_list[0]).shape

        # Initialize paths
        forward_frame_paths = []
        forward_mask_paths = []
        backward_frame_paths = []
        backward_mask_paths = []

        # Initialize frames and masks
        previous_forward_frame = np.full((height, width, 3), fill_color, dtype=np.uint8)
        previous_forward_mask = np.full((height, width), 0, dtype=np.uint32)
        previous_backward_frame = np.full((height, width, 3), fill_color, dtype=np.uint8)
        previous_backward_mask = np.full((height, width), num_frames - 1, dtype=np.uint32)

        # Create forward frames and masks
        forward_frame_paths, forward_mask_paths = self._process_frames_and_masks_on_disk(
            frame_path_list,
            mask_path_list,
            previous_forward_frame,
            previous_forward_mask,
            output_dir,
            "forward",
            range(num_frames),
            leading_zeros=leading_zeros,
        )

        # Create backward frames and masks
        backward_frame_paths, backward_mask_paths = self._process_frames_and_masks_on_disk(
            frame_path_list,
            mask_path_list,
            previous_backward_frame,
            previous_backward_mask,
            output_dir,
            "backward",
            range(num_frames - 1, -1, -1),
            leading_zeros=leading_zeros,
        )

        return forward_frame_paths, forward_mask_paths, backward_frame_paths, backward_mask_paths

    def _process_frames_and_masks_on_disk(
        self,
        frame_path_list,
        mask_path_list,
        previous_frame,
        previous_mask,
        output_dir,
        direction,
        frame_range,
        leading_zeros: int = 8,
    ):
        frame_paths = [""] * len(frame_range)
        mask_paths = [""] * len(frame_range)

        for i in frame_range:
            current_frame = cv2.imread(frame_path_list[i])
            current_mask = cv2.imread(mask_path_list[i], cv2.IMREAD_GRAYSCALE)

            processed_frame = np.where(current_mask[..., None] == 0, current_frame, previous_frame)
            processed_mask = np.where(current_mask == 0, i, previous_mask)

            frame_path = f"{output_dir}/{direction}_frame_{str(i).zfill(leading_zeros)}.png"
            cv2.imwrite(frame_path, processed_frame)
            # Store as a numpy array since entries have a different data type
            mask_path = f"{output_dir}/{direction}_mask_{str(i).zfill(leading_zeros)}.npy"
            np.save(mask_path, processed_mask)

            frame_paths[i] = frame_path
            mask_paths[i] = mask_path

            previous_frame = processed_frame
            previous_mask = processed_mask

        return frame_paths, mask_paths

    def create_merged_frame_list_on_disk(
        self,
        frame_path_list: List[str],
        mask_path_list: List[str],
        forward_frame_path_list: List[str],
        forward_mask_path_list: List[str],
        backward_frame_path_list: List[str],
        backward_mask_path_list: List[str],
        output_dir: str,
        leading_zeros: int = 8,
    ):
        """
        Merge frames based on the nearest unoccluded pixel, storing the results on disk.

        Creates a merged frame list where each frame is created as follows:
        When the original mask is 0, use the original image.
        Otherwise, use the formula:
        merged_frame = ((i - forward_mask) * forward_frame + (backward_mask - i) * backward_frame) / (backward_mask - forward_mask)

        Args:
            frame_path_list: List of paths to frames representing the video.
            mask_path_list: List of paths to masks corresponding to each frame.
            forward_frame_path_list: List of paths to forward frames.
            forward_mask_path_list: List of paths to forward masks.
            backward_frame_path_list: List of paths to backward frames.
            backward_mask_path_list: List of paths to backward masks.
            output_dir: Directory where the output merged frames will be stored.
            leading_zeros: Number of leading zeros before frame index.

        Returns:
            merged_frame_paths: List of paths to merged frames.
        """
        num_frames = len(frame_path_list)
        merged_frame_paths = []

        for i in range(num_frames):
            current_frame = cv2.imread(frame_path_list[i])
            current_mask = cv2.imread(mask_path_list[i], cv2.IMREAD_GRAYSCALE)
            forward_frame = cv2.imread(forward_frame_path_list[i])
            forward_mask = np.load(forward_mask_path_list[i])
            backward_frame = cv2.imread(backward_frame_path_list[i])
            backward_mask = np.load(backward_mask_path_list[i])

            merged_frame = np.where(
                current_mask[..., None] == 0,
                current_frame,
                ((i - forward_mask[..., None]) * forward_frame + (backward_mask[..., None] - i) * backward_frame)
                / (backward_mask[..., None] - forward_mask[..., None] + np.finfo(float).eps),
            )

            merged_frame_path = f"{output_dir}/merged_frame_{str(i).zfill(leading_zeros)}.png"
            cv2.imwrite(merged_frame_path, merged_frame.astype(np.uint8))
            merged_frame_paths.append(merged_frame_path)

        return merged_frame_paths

    def create_forward_backward_frames_and_masks(
        self, frames: List[np.ndarray], masks: List[np.ndarray], fill_color: Tuple[int, int, int]
    ):
        """
        Creates forward and backward frames and masks for a sequence of frames and masks.

        Args:
            frames: List of frames representing the video.
            masks: List of masks corresponding to each frame.
                The mask is either 0 (the pixel is good to use) or 255 (the pixel may be occluded).
            fill_color: The color that gets filled in the occluded region of the
                first and last frame if no other viable pixel was found.

        Returns:
            forward_frame_list: List of forward frames.
            forward_mask_list: List of forward masks.
            backward_frame_list: List of backward frames.
            backward_mask_list: List of backward masks.
        """
        num_frames = len(frames)
        height, width, _ = frames[0].shape

        # Initialize forward and backward frames and masks
        forward_frame_list = []
        forward_mask_list = []
        backward_frame_list = []
        backward_mask_list = []

        # Initialize frames and masks
        previous_forward_frame = np.full((height, width, 3), fill_color, dtype=np.uint8)
        previous_forward_mask = np.full((height, width), 0, dtype=np.uint32)
        previous_backward_frame = np.full((height, width, 3), fill_color, dtype=np.uint8)
        previous_backward_mask = np.full((height, width), num_frames - 1, dtype=np.uint32)

        # Create forward frames and masks
        forward_frame_list, forward_mask_list = self._process_frames_and_masks(
            frames, masks, previous_forward_frame, previous_forward_mask, range(num_frames)
        )

        # Create backward frames and masks
        backward_frame_list, backward_mask_list = self._process_frames_and_masks(
            frames, masks, previous_backward_frame, previous_backward_mask, range(num_frames - 1, -1, -1)
        )

        return forward_frame_list, forward_mask_list, backward_frame_list, backward_mask_list

    def _process_frames_and_masks(self, frames, masks, previous_frame, previous_mask, frame_range):
        frame_list = [np.ndarray(0)] * len(frame_range)
        mask_list = [np.ndarray(0)] * len(frame_range)

        for i in frame_range:
            current_frame = frames[i]
            current_mask = masks[i]

            processed_frame = np.where(current_mask[..., None] == 0, current_frame, previous_frame)
            processed_mask = np.where(current_mask == 0, i, previous_mask)

            frame_list[i] = processed_frame
            mask_list[i] = processed_mask

            previous_frame = processed_frame
            previous_mask = processed_mask

        return frame_list, mask_list

    def create_merged_frame_list(
        self,
        frames: List[np.ndarray],
        masks: List[np.ndarray],
        forward_frame_list: List[np.ndarray],
        forward_mask_list: List[np.ndarray],
        backward_frame_list: List[np.ndarray],
        backward_mask_list: List[np.ndarray],
    ):
        """
        Merge frames based on the nearest unoccluded pixel.

        Creates a merged frame list where each frame is created as follows:
        When the original mask is 0, use the original image.
        Otherwise, use the formula:
        merged_frame = ((i - forward_mask) * forward_frame + (backward_mask - i) * backward_frame) / (backward_mask - forward_mask)
        leading_zeros: Number of leading zeros before frame index.

        Args:
            frames: List of frames representing the video.
            masks: List of masks corresponding to each frame.
            forward_frame_list: List of forward frames.
            forward_mask_list: List of forward masks.
            backward_frame_list: List of backward frames.
            backward_mask_list: List of backward masks.

        Returns:
            merged_frame_list: List of merged frames.
        """
        num_frames = len(frames)
        merged_frame_list = []

        for i in range(num_frames):
            current_frame = frames[i]
            current_mask = masks[i]
            forward_frame = forward_frame_list[i]
            forward_mask = forward_mask_list[i]
            backward_frame = backward_frame_list[i]
            backward_mask = backward_mask_list[i]

            merged_frame = np.where(
                current_mask[..., None] == 0,
                current_frame,
                ((i - forward_mask[..., None]) * forward_frame + (backward_mask[..., None] - i) * backward_frame)
                / (backward_mask[..., None] - forward_mask[..., None] + np.finfo(float).eps),
            )

            merged_frame_list.append(merged_frame.astype(np.uint8))

        return merged_frame_list

    def remove_occlusions(self, frame_list: List[np.ndarray], mask_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Removes occlusions indicated by the masks.

        For the occluded pixels the method merges the previous unoccluded pixel
        with the next unoccluded one.

        Args:
            frame_list: List of frames in cv2 image format.
            mask_list: List of masks in cv2 format.

        Returns:
            List of frames where the parts of the frame indicated
            by the masks (occlusions) have been removed.
        """
        fill_color = self.params.get("fill_color", (255, 255, 255))
        forward_frame_list, forward_mask_list, backward_frame_list, backward_mask_list = (
            self.create_forward_backward_frames_and_masks(frame_list, mask_list, fill_color=fill_color)
        )
        merged_frame_list = self.create_merged_frame_list(
            frame_list, mask_list, forward_frame_list, forward_mask_list, backward_frame_list, backward_mask_list
        )
        return merged_frame_list

    def remove_occlusions_on_disk(
        self, frame_path_list: List[str], mask_path_list: List[str], output_dir: str
    ) -> List[str]:
        """
        Removes occlusions indicated by the masks.

        For the occluded pixels the method merges the previous unoccluded pixel
        with the next unoccluded one.

        Args:
            frame_path_list: List of paths to frames representing the video.
            mask_path_list: List of paths to masks corresponding to each frame.
            output_dir: Directory where the output merged frames will be stored.

        Returns:
            List of frame paths where the parts of the frame indicated
            by the masks (occlusions) have been removed.
        """
        fill_color = self.params.get("fill_color", (255, 255, 255))
        delete_intermediate_data = self.params.get("delete_intermediate_data", True)

        forward_frame_path_list, forward_mask_path_list, backward_frame_path_list, backward_mask_path_list = (
            self.create_forward_backward_frames_and_masks_on_disk(
                frame_path_list, mask_path_list, output_dir, fill_color=fill_color
            )
        )
        merged_frame_path_list = self.create_merged_frame_list_on_disk(
            frame_path_list,
            mask_path_list,
            forward_frame_path_list,
            forward_mask_path_list,
            backward_frame_path_list,
            backward_mask_path_list,
            output_dir,
        )

        if delete_intermediate_data:
            self.remove_files_from_disk(forward_frame_path_list)
            self.remove_files_from_disk(forward_mask_path_list)
            self.remove_files_from_disk(backward_frame_path_list)
            self.remove_files_from_disk(backward_mask_path_list)

        return merged_frame_path_list

"""Color Correction."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# Original code: https://github.com/CraGL/timelapse/blob/master/1%20preprocessing/s1_whole_sequence_colorshift/colorshift_filter.cpp


class VideoFilter:
    """
    This class is a base class for video filters.
    """

    def __init__(self):
        self.output = None

    def set_output(self, out: VideoFilter):
        """
        Sets the output filter.

        Args:
            out: The next filter in the pipeline.
        """
        self.output = out

    def next_frame(self, frame: np.ndarray):
        """
        Processes the next frame.

        Args:
            frame: The input frame.

        Returns:
            np.ndarray: The processed frame.
        """
        result = np.zeros_like(frame)
        success = self.do_next_frame(frame, result)
        if not success:
            return result

        if self.output:
            return self.output.next_frame(result)
        else:
            return result

    def do_next_frame(self, input_frame, output_frame):
        """
        Processes the input frame and produces the output frame.

        Args:
            input_frame (np.ndarray): The input frame.
            output_frame (np.ndarray): The output frame.
        """
        raise NotImplementedError("Subclasses should implement this method")


class ColorShift(VideoFilter):
    """
    This class performs color shift correction on video frames.
    """

    def __init__(self, params: Dict[str, Any]):
        """
        Initializes the ColorShift filter.

        Args:
            params: Parameters for the filter.
        """
        super().__init__()
        self.percent = params["Percent"]
        self.threshold = params["Threshold"]
        self.fixed_frame = None

    def do_next_frame(self, input_frame: np.ndarray, output_frame: np.ndarray):
        """
        Processes the input frame and produces the output frame.

        Args:
            input_frame: The input frame.
            output_frame: The output frame.

        Returns:
            bool: True if the frame was processed successfully, False otherwise.
        """
        if self.fixed_frame is None:
            self.fixed_frame = input_frame.copy()
            output_frame[:] = self.fixed_frame
            return True

        current_diff = self.diff(self.fixed_frame, input_frame)
        height, width = input_frame.shape[:2]

        # Binary mask for differences exceeding self.threshold
        temp = cv2.threshold(current_diff, self.threshold, 1.0, cv2.THRESH_BINARY)[1]
        # count the non zero element, and do not change tframe corresponding to the frame whose number > self.percent
        if np.count_nonzero(temp) > self.percent * height * width:
            self.color_shift_recover(input_frame, output_frame, self.fixed_frame)
        else:
            output_frame[:] = input_frame

        return True

    def to_lab(self, bgr: np.ndarray):
        """
        Converts a BGR image to a normalized float32 format.

        Args:
            bgr: The input BGR image.

        Returns:
            np.ndarray: The normalized image.
        """
        bgr32 = bgr.astype(np.float32) / 255.0
        return bgr32

    def diff(self, mat1: np.ndarray, mat2: np.ndarray):
        """
        Computes the difference between two images in color space.

        Args:
            mat1: The first image.
            mat2: The second image.

        Returns:
            np.ndarray: The difference image.
        """
        mat1_lab = self.to_lab(mat1)
        mat2_lab = self.to_lab(mat2)
        diff = cv2.absdiff(mat1_lab, mat2_lab)

        # Optimization of https://github.com/CraGL/timelapse\
        # /blob/master/1%20preprocessing/s1_whole_sequence_colorshift\
        # /colorshift_filter.cpp#L212
        diff = np.sqrt(np.sum(diff**2, axis=2)) / 1.8
        return (diff * 255).astype(np.uint8)

    def color_shift_recover(self, input_frame: np.ndarray, output_frame: np.ndarray, fixed_frame: np.ndarray):
        """
        Applies color shift recovery to the input frame.

        Args:
            input_frame: The input frame.
            output_frame: The output frame.
            fixed_frame: The reference frame.
        """
        height, width = input_frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        self.get_mask(fixed_frame, input_frame, mask)

        channels = cv2.split(fixed_frame)
        new_channels = cv2.split(input_frame)
        recovered_channels = []

        for old_channel, new_channel in zip(channels, new_channels):
            recovered_channel = self.solve_lsm(old_channel, new_channel, mask)
            recovered_channels.append(recovered_channel)

        output_frame[:] = cv2.merge(recovered_channels)
        self.fixed_frame = output_frame.copy()

    def get_mask(self, before, after, mask):
        """
        Generates a mask based on color differences.

        Those pixels get masked where the difference between the frames
        are between the 8th and 4th percentile.

        Args:
            before (np.ndarray): The reference frame.
            after (np.ndarray): The current frame.
            mask (np.ndarray): The output mask.
        """
        before = before.astype(np.float64) / 255.0
        after = after.astype(np.float64) / 255.0
        diff = np.sum((before - after) ** 2, axis=2)

        diff_list = [(diff[i, j], i, j) for i in range(diff.shape[0]) for j in range(diff.shape[1])]
        diff_list.sort()

        # To get the indices between the 8th and 4th percentile
        N = len(diff_list)
        for _, i, j in diff_list[N // 8 : N // 4]:
            mask[i, j] = 1

    def solve_lsm(self, old_img: np.ndarray, new_img: np.ndarray, mask: np.ndarray):
        """
        Solves the least squares problem for color correction.

        Args:
            old_img: The reference image.
            new_img: The current image.
            mask: The mask for selected pixels.

        Returns:
            np.ndarray: The corrected image.
        """
        old_img = old_img.astype(np.float64) / 255.0
        new_img = new_img.astype(np.float64) / 255.0

        mask_indices = np.where(mask == 1)
        count = len(mask_indices[0])

        new_sum = np.sum(new_img[mask_indices])
        old_sum = np.sum(old_img[mask_indices])
        new_sum_square = np.sum(new_img[mask_indices] ** 2)
        old_new_sum = np.sum(new_img[mask_indices] * old_img[mask_indices])

        M = csr_matrix([[count, new_sum], [new_sum, new_sum_square]])
        N = np.array([old_sum, old_new_sum])

        # Solve MX = N
        X = spsolve(M, N)

        recovered = old_img.copy()
        img_diff_mask = np.where(abs(old_img - new_img) >= 0)
        recovered[img_diff_mask] = X[0] + X[1] * new_img[img_diff_mask]
        recovered = np.clip(recovered, 0, 1)

        return (recovered * 255).astype(np.uint8)


def process_video(input_path: str, output_path: str, filter_obj: VideoFilter, output_vide_format: str = "DIVX"):
    """
    Processes a video using the specified filter and saves the output.

    Usage:
    ```.py
        params = {"Percent": 0.2, "Threshold": 30}
        color_shift = ColorShift(params)

        process_video('input_video.mp4', 'output_video.avi', color_shift)
    ```

    Args:
        input_path: Path to the input video file.
        output_path: Path to the output video file.
        filter_obj: The filter to apply to the video.
        output_vide_format: cv2 format.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*output_vide_format), fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = filter_obj.next_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()

"""Color Correction."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from painvidpro.timelapse.utils import color_shift_recover, diff


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
        success, result = self.do_next_frame(frame)
        if not success:
            return np.zeros_like(frame)

        if self.output:
            return self.output.next_frame(result)
        else:
            return result

    def do_next_frame(self, input_frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Processes the input frame and produces the output frame.

        Args:
            input_frame: The input frame.

        Returns:
            Sucess as a bool and the next_frame as a np.ndarray.
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

    def do_next_frame(self, input_frame: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        Processes the input frame and produces the output frame.

        Args:
            input_frame: The input frame.

        Returns:
            bool: True if the frame was processed successfully, False otherwise.
            next_frame: an np.ndarray
        """
        if self.fixed_frame is None:
            self.fixed_frame = input_frame.copy()
            return True, self.fixed_frame

        current_diff = diff(self.fixed_frame, input_frame)
        height, width = input_frame.shape[:2]

        # Binary mask for differences exceeding self.threshold
        temp = cv2.threshold(current_diff, self.threshold, 1.0, cv2.THRESH_BINARY)[1]
        # count the non zero element, and do not change tframe corresponding to the frame whose number > self.percent
        if np.count_nonzero(temp) > self.percent * height * width:
            recovered_frame = color_shift_recover(input_frame, self.fixed_frame)
            # There seems to be some uncertanty in the original code about these lines:
            output_frame = recovered_frame
            self.fixed_frame = output_frame.copy()
        else:
            output_frame = input_frame

        return True, output_frame


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
    numb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*output_vide_format), fps, (frame_width, frame_height))

    with tqdm(total=numb_frames, desc="Frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = filter_obj.next_frame(frame)
            out.write(processed_frame)
            pbar.update(1)

    cap.release()
    out.release()

"""Fixed frame number for the relevant sequence detection."""

from typing import List, Union

import cv2
import numpy as np

from painvidpro.sequence_detection.base import BaseSequence, SequenceDetectionBase
from painvidpro.video_processing.utils import video_capture_context


class SequenceDetectionFixed(SequenceDetectionBase):
    def __init__(self):
        """Class to detect sequences."""
        super().__init__()
        self.set_default_parameters()

    def set_default_parameters(self):
        self.params = {
            "start_offset": 100,
            "end_offset": 100,
        }

    def detect_sequences(self, frame_list: List[np.ndarray]) -> List[BaseSequence]:
        """
        Detects frame sequences which contain painting content.

        Returns a List of BaseSequence.
        A BaseSequence is defined by a start index and end frame index.

        The start index is the defined start offset.
        The end index is the number of frames minus the end offset.
        If the number of frames is smaller then the two offsets combines returns 0 and
        nubmer frames as sequence indices.

        Args:
            frame_list: List of frames in cv2 image format.

        Returns:
            List of BaseSequence objects, each containing a sequence.
        """
        start_offset = self.params.get("start_offset", 100)
        end_offset = self.params.get("end_offset", 100)
        num_frames = len(frame_list)
        desc = "Video content."
        if num_frames > (start_offset + end_offset):
            return [BaseSequence(start_idx=start_offset, end_idx=(num_frames - end_offset), desc=desc)]
        else:
            return [BaseSequence(0, num_frames, desc=desc)]

    def detect_sequences_on_disk(self, frame_path: Union[str, List[str]]) -> List[BaseSequence]:
        """
        Detects frame sequences which contain painting content.

        Returns a List of BaseSequence.
        A BaseSequence is defined by a start index and end frame index.

        The start index is the defined start offset.
        The end index is the number of frames minus the end offset.
        If the number of frames is smaller then the two offsets combines returns 0 and
        nubmer frames as sequence indices.

        Args:
            frame_path:
                If it is a string than it gets interpreted as a cv2.VideoCapture filepath, this
                can be either an open video file or image file sequence.
                If it is a List of strings it gets interpredted as an image file sequence.

        Returns:
            List of BaseSequence objects, each containing a sequence.

        Raises:
            ValueError if frame_path can not be opened.
        """
        start_offset = self.params.get("start_offset", 100)
        end_offset = self.params.get("end_offset", 100)
        if isinstance(frame_path, List):
            num_frames = len(frame_path)
        else:
            with video_capture_context(video_path=frame_path) as cap:
                if not cap.isOpened():
                    raise ValueError(f"Was not able to read from {frame_path}.")
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        desc = "Video content."
        if num_frames > (start_offset + end_offset):
            return [BaseSequence(start_idx=start_offset, end_idx=(num_frames - end_offset), desc=desc)]
        else:
            return [BaseSequence(0, num_frames, desc=desc)]

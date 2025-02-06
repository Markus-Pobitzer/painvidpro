"""Base class for the relevant sequence detection."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np


@dataclass
class BaseSequence:
    """Stores a sequence of Frames."""

    start_idx: int
    end_idx: int
    desc: str = ""


class SequenceDetectionBase:
    def __init__(self):
        """Base class to detect sequences."""
        self.params: Dict[str, Any] = {}

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        self.params.update(params)
        return True, ""

    def set_default_parameters(self):
        raise NotImplementedError("This method should be implemented by the child class.")

    def detect_sequences(self, frame_list: List[np.ndarray]) -> List[BaseSequence]:
        """
        Detects frame sequences which contain painting content.

        Returns a List of BaseSequence.
        A BaseSequence is defined by a start index and end frame index.
        The function returns BaseSequences ordered by the start frame index.

        Args:
            frame_list: List of frames in cv2 image format.

        Returns:
            List of BaseSequence objects, each containing a sequence.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

    def detect_sequences_on_disk(self, frame_path: Union[str, List[str]]) -> List[BaseSequence]:
        """
        Detects frame sequences which contain painting content.

        Returns a List of BaseSequence.
        A BaseSequence is defined by a start index and end frame index.
        The function returns BaseSequences ordered by the start frame index.

        Args:
            frame_path:
                If it is a string than it gets interpreted as a cv2.VideoCapture filepath, this
                can be either an open video file or image file sequence.
                If it is a List of strings it gets interpredted as an image file sequence.

        Returns:
            List of BaseSequence objects, each containing a sequence.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

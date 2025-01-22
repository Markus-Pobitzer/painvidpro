"""Base class for the Keyframe detection."""

from typing import Any, Dict, List, Tuple

import numpy as np


class KeyframeDetectionBase:
    def __init__(self):
        """Base class to detect keyframes."""
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

    def detect_keyframes(self, frame_list: List[np.ndarray]) -> List[List[int]]:
        """
        Detects frames which have no occlusions.

        Args:
            frame_list: List of frames in cv2 image format.

        Returns:
            List of frame sequences. Each sequence corresponds to one Keyframe.
            Note that several frames can correspond to one Keyframe if there
            is no changes between them.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

    def detect_keyframes_on_disk(self, frame_path_list: List[str]) -> List[List[int]]:
        """
        Detects frames which have no occlusions.

        Images are loaded directly from disk

        Args:
            frame_list: List of frames in cv2 image format.

        Returns:
            List of frame sequences. Each sequence corresponds to one Keyframe.
            Note that several frames can correspond to one Keyframe if there
            is no changes between them.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

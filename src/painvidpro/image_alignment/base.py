"""Base class for the imge alignment."""

from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np


class AlignmentStatus(Enum):
    SUCCESS = 1
    FAILURE = 2
    NOT_ENOUGH_MATCHES = 3
    NOT_ENOUGH_GOOD_MATCHES = 4
    HOMOGRAPHY_FAILED = 5


class ImageAlignmentBase:
    def __init__(self):
        """Base class to align images."""
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

    def align_images(
        self, ref_frame: np.ndarray, frame_list: List[np.ndarray]
    ) -> List[Tuple[AlignmentStatus, Any, Any, Any, Any, Any]]:
        """
        Aligns the frames of frame_list to the ref_frame.

        Args:
            ref_frame: The reference frame.
            frame_list: List of frames in cv2 image format.

        Returns:
            For each frame in frame list a Tuple with following entries:
                AlignmentStatus: The AlignmentStatus.
                aligned_frame: The aligned frame.
                mask: Mask where the alignment produced empty space.
                keypoints_ref: The keypoints in the reference image.
                keypoints_frame: The keypoints in the frame.
                matches: The comptued matches.
            If the AlignmentStatus of an entry is not AlignmentStatus.SUCCESS, the other entries in the
            Tuple are undefined.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

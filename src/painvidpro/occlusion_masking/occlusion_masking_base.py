"""Base class for the Occlusion Masking."""

from typing import Any, Dict, List, Tuple

import numpy as np


class OcclusionMaskingBase:
    def __init__(self):
        self.params = {}

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict witht the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        self.params.update(params)
        return True, ""

    def set_default_parameters(self):
        raise NotImplementedError("This method should be implemented by the child class.")

    def compute_mask_list(self, frame_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Computes occlusion masks based on the input frames.

        Args:
            frame_list: List of frames in cv2 image format.

        Returns:
            List of masks, one for each input frame where an occlusion may exist.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

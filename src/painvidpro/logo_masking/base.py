"""Base class for the Logo Masking."""

from typing import Any, Dict, List, Tuple

import numpy as np


class LogoMaskingBase:
    def __init__(self):
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

    def offload_model(self):
        """Offloads the model to CPU, no effect if methdod has no model."""
        pass

    def set_default_parameters(self):
        raise NotImplementedError("This method should be implemented by the child class.")

    def compute_mask_list(self, frame_list: List[np.ndarray], offload_model: bool = True) -> List[np.ndarray]:
        """
        Computes logo masks based on the input frames.

        Args:
            frame_list: List of frames in cv2 image format.
            offload_model: Loads the internal to CPU after usage, only if applicable.

        Returns:
            List of masks, one for each input frame where a Logo may exist.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

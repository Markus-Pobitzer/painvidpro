"""Base class for the object detection."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np


class ObjectDetectionBase:
    def __init__(self):
        """Base class to detect objects."""
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

    def offload_model(self):
        """Offloads the model to CPU, no effect if methdod has no model."""
        raise NotImplementedError("This method should be implemented by the child class.")

    def detect_objects(
        self, frame_list: Union[List[np.ndarray], List[str]], offload_model: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """
        Detects objets from frames.

        Returns a List of detected objects, indicating for each frame if the object is present.
        The desired objects should be set as parameters.

        Args:
            frame_list: List of frames in cv2 image format or paths.

        Returns:
            List of Lists indicatinig detected objects. For each image a List of dicts.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

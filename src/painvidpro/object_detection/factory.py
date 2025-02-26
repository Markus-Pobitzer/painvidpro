"""Factory for detecting objects of frames."""

from typing import Any, Dict

from painvidpro.object_detection.base import ObjectDetectionBase
from painvidpro.object_detection.grounding_dino import ObjectDetectionGroundingDino


class ObjectDetectionFactory:
    @staticmethod
    def build(algorithm: str, params: Dict[str, Any]) -> ObjectDetectionBase:
        """Builds an ObjectDetection Algorithm.

        Args:
            algorithm: The Algorithm as a string.
            params: The parameters as a dict.

        Returns:
            An ObjectDetection Algorithm.

        Raises:
            ValueError if the alogrithm is unknown or the algorithm can not
            be build.
        """
        instance: ObjectDetectionBase
        if algorithm == "ObjectDetectionGroundingDino":
            instance = ObjectDetectionGroundingDino()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

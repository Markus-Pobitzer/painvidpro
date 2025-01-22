"""Factory for detecting keyframes."""

from typing import Any, Dict

from painvidpro.keyframe_detection.base import KeyframeDetectionBase
from painvidpro.keyframe_detection.frame_diff import KeyframeDetectionFrameDiff


class KeyframeDetectionFactory:
    @staticmethod
    def build(algorithm: str, params: Dict[str, Any]) -> KeyframeDetectionBase:
        """Builds an KeyframeDetection Algorithm.

        Args:
            algorithm: The Algorithm as a string.
            params: The parameters as a dict.

        Returns:
            An KeyframeDetection Algorithm.

        Raises:
            ValueError if the alogrithm is unknown or the algorithm can not
            be build.
        """
        instance: KeyframeDetectionBase
        if algorithm == "KeyframeDetectionFrameDiff":
            instance = KeyframeDetectionFrameDiff()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

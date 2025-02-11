"""Factory for detecting sequences of frames."""

from typing import Any, Dict

from painvidpro.sequence_detection.base import SequenceDetectionBase
from painvidpro.sequence_detection.fixed import SequenceDetectionFixed
from painvidpro.sequence_detection.grounding_dino import SequenceDetectionGroundingDino


class SequenceDetectionFactory:
    @staticmethod
    def build(algorithm: str, params: Dict[str, Any]) -> SequenceDetectionBase:
        """Builds an SequenceDetection Algorithm.

        Args:
            algorithm: The Algorithm as a string.
            params: The parameters as a dict.

        Returns:
            An SequenceDetection Algorithm.

        Raises:
            ValueError if the alogrithm is unknown or the algorithm can not
            be build.
        """
        instance: SequenceDetectionBase
        if algorithm == "SequenceDetectionFixed":
            instance = SequenceDetectionFixed()
        elif algorithm == "SequenceDetectionGroundingDino":
            instance = SequenceDetectionGroundingDino()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

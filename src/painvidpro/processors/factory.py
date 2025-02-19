"""Factory for processing videos."""

from typing import Any, Dict

from painvidpro.processors.base import ProcessorBase
from painvidpro.processors.keyframe import ProcessorKeyframe
from painvidpro.processors.loomis_keyframe import ProcessorLoomisKeyframe


class ProcessorsFactory:
    @staticmethod
    def build(processor: str, params: Dict[str, Any]) -> ProcessorBase:
        """Builds an Processor pipeline.

        Args:
            processor: The processor name as a string.
            params: The parameters as a dict.

        Returns:
            An Processor pipeline.

        Raises:
            ValueError if the alogrithm is unknown or the algorithm can not
            be build.
        """
        instance: ProcessorBase
        if processor == "ProcessorKeyframe":
            instance = ProcessorKeyframe()
        elif processor == "ProcessorLoomisKeyframe":
            instance = ProcessorLoomisKeyframe()
        else:
            raise ValueError(f"Unknown processor: {processor}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

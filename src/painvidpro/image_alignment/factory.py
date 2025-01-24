"""Factory for image alignment."""

from typing import Any, Dict

from painvidpro.image_alignment.base import ImageAlignmentBase
from painvidpro.image_alignment.light_glue import ImageAlignmentLightGlue
from painvidpro.image_alignment.orb import ImageAlignmentOrb


class ImageAlignmentFactory:
    @staticmethod
    def build(algorithm: str, params: Dict[str, Any]) -> ImageAlignmentBase:
        """Builds a ImageAlignment Algorithm.

        Args:
            algorithm: The Algorithm as a string.
            params: The parameters as a dict.

        Returns:
            An ImageAlignment Algorithm.

        Raises:
            ValueError if the alogrithm is unknown or the algorithm can not
            be build.
        """
        instance: ImageAlignmentBase
        if algorithm == "ImageAlignmentOrb":
            instance = ImageAlignmentOrb()
        elif algorithm == "ImageAlignmentLightGlue":
            instance = ImageAlignmentLightGlue()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

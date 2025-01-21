"""Factory for the Occlusion Masking."""

from typing import Any, Dict

from painvidpro.occlusion_masking.accumulate_diff import OcclusionMaskingAccumulateDiff
from painvidpro.occlusion_masking.base import OcclusionMaskingBase
from painvidpro.occlusion_masking.frame_diff import OcclusionMaskingFrameDiff


class OcclusionMaskingFactory:
    @staticmethod
    def build(algorithm: str, params: Dict[str, Any]) -> OcclusionMaskingBase:
        """Builds an OcclusionMasking Algorithm.

        Args:
            algorithm: The Algorithm as a string.
            params: The parameters as a dict.

        Returns:
            An OcclusionMasking Algorithm.

        Raises:
            ValueError if the alogrithm is unknown or the algorithm can not
            be build.
        """
        instance: OcclusionMaskingBase
        if algorithm == "OcclusionMaskingFrameDiff":
            instance = OcclusionMaskingFrameDiff()
        elif algorithm == "OcclusionMaskingAccumulateDiff":
            instance = OcclusionMaskingAccumulateDiff()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

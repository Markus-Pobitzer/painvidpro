"""Factory for the Occlusion Masking."""

from typing import Any, Dict

from painvidpro.occlusion_masking.accumulate_diff import OcclusionMaskingAccumulateDiff
from painvidpro.occlusion_masking.base import OcclusionMaskingBase
from painvidpro.occlusion_masking.dav2 import OcclusionMaskingDAV2
from painvidpro.occlusion_masking.frame_diff import OcclusionMaskingFrameDiff
from painvidpro.occlusion_masking.inspyrenet import OcclusionMaskingInSPyReNet
from painvidpro.occlusion_masking.rmbg import OcclusionMaskingRMBG
from painvidpro.occlusion_masking.sam3 import OcclusionMaskingSAM3


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
        elif algorithm == "OcclusionMaskingRMBG":
            instance = OcclusionMaskingRMBG()
        elif algorithm == "OcclusionMaskingDAV2":
            instance = OcclusionMaskingDAV2()
        elif algorithm == "OcclusionMaskingInSPyReNet":
            instance = OcclusionMaskingInSPyReNet()
        elif algorithm == "OcclusionMaskingSAM3":
            instance = OcclusionMaskingSAM3()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

"""Factory for removing occlusions."""

from typing import Any, Dict

from painvidpro.occlusion_removing.base import OcclusionRemovingBase
from painvidpro.occlusion_removing.frame_merging import OcclusionRemovingFrameMerging
from painvidpro.occlusion_removing.lama_inpainting import OcclusionRemovingLamaInpainting


class OcclusionRemovingFactory:
    @staticmethod
    def build(algorithm: str, params: Dict[str, Any]) -> OcclusionRemovingBase:
        """Builds an Occlusionremoving Algorithm.

        Args:
            algorithm: The Algorithm as a string.
            params: The parameters as a dict.

        Returns:
            An OcclusionRemoving Algorithm.

        Raises:
            ValueError if the alogrithm is unknown or the algorithm can not
            be build.
        """
        instance: OcclusionRemovingBase
        if algorithm == "OcclusionRemovingFrameMerging":
            instance = OcclusionRemovingFrameMerging()
        elif algorithm == "OcclusionRemovingLamaInpainting":
            instance = OcclusionRemovingLamaInpainting()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

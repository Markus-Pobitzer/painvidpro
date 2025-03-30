"""Factory for the Logo Masking."""

from typing import Any, Dict

from painvidpro.logo_masking.base import LogoMaskingBase
from painvidpro.logo_masking.easy_ocr import LogoMaskingEasyOCR
from painvidpro.logo_masking.no_changes import LogoMaskingNoChanges


class LogoMaskingFactory:
    @staticmethod
    def build(algorithm: str, params: Dict[str, Any]) -> LogoMaskingBase:
        """Builds an LogoMasking Algorithm.

        Args:
            algorithm: The Algorithm as a string.
            params: The parameters as a dict.

        Returns:
            An LogoMasking Algorithm.

        Raises:
            ValueError if the alogrithm is unknown or the algorithm can not
            be build.
        """
        instance: LogoMaskingBase
        if algorithm == "LogoMaskingNoChanges":
            instance = LogoMaskingNoChanges()
        elif algorithm == "LogoMaskingEasyOCR":
            instance = LogoMaskingEasyOCR()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        success, error_message = instance.set_parameters(params)
        if not success:
            raise ValueError(f"Error setting parameters: {error_message}")

        return instance

"""Implementation of OcclusionMaskingNoChnges."""

from typing import Any, Dict, List, Optional, Tuple

import easyocr
import numpy as np

from painvidpro.logo_masking.base import LogoMaskingBase


class LogoMaskingEasyOCR(LogoMaskingBase):
    def __init__(self):
        """Class to compute logo masks."""
        super().__init__()
        self.set_default_parameters()
        self._reader: Optional[easyocr.Reader] = None

    @property
    def reader(self) -> easyocr.Reader:
        if self._reader is None:
            raise RuntimeError(
                ("OCR Reader not instanciated. Make sure to call " "set_parameters to laod the reader first.")
            )
        return self._reader

    def set_default_parameters(self):
        self.params = {"device": "cuda", "languages": "en"}

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        self.params.update(params)

        device = self.params["device"]
        use_gpu = False if device == "cpu" else True
        try:
            languages = self.params.get("languages", "en").split(",")
            self._reader = easyocr.Reader(languages, gpu=use_gpu)
        except Exception as e:
            return False, str(e)
        return True, ""

    def compute_mask_list(self, frame_list: List[np.ndarray]) -> List[np.ndarray]:
        """Logo mask indicating Text.

        All frames get analyzed and checked which areas contain text.

        Args:
            frame_list: List of frames in cv2 image format.

        Returns:
            List with a binary mask for each image.
        """
        ret: List[np.ndarray] = []
        for img in frame_list:
            # ret.append(self.reader.detect(img))
            ret.append(self.reader.readtext(img))
        return ret

"""Occlusion Masking."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from transformers import Pipeline, pipeline

from painvidpro.occlusion_masking.base import OcclusionMaskingBase


class OcclusionMaskingDAV2(OcclusionMaskingBase):
    def __init__(self):
        """Class to detect sequences."""
        super().__init__()
        self._pipe: Optional[Pipeline] = None
        self.set_default_parameters()

    @property
    def pipe(self) -> Pipeline:
        if self._pipe is None:
            raise RuntimeError(
                (
                    "Model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._pipe

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        self.params.update(params)

        model_id = self.params["model_id"]
        device = self.params["device"]
        self._pipe = pipeline(task="depth-estimation", model=model_id, device=device)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "model_id": "depth-anything/Depth-Anything-V2-Base-hf",
            "device": "cuda",
            "batch_size": 1,
            "depth_threshold": 0.25,
            "convert_input_from_bgr_to_rgb": True,
        }

    def _prepare_img(self, img: np.ndarray) -> Image.Image:
        """Converts a numpy array image to PIL."""
        convert_input_from_bgr_to_rgb = self.params.get("convert_input_from_bgr_to_rgb", True)
        if convert_input_from_bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

    def compute_mask_list(
        self, frame_list: Union[List[np.ndarray], np.ndarray], offload_model: bool = True
    ) -> List[np.ndarray]:
        """
        Computes occlusion masks based on the input frames, using the briaai/RMBG model.

        Args:
            frame_list: List of frames in cv2 image format.
                Can also be a np.ndarray with shape [B, H, W, C], where B is the number
                of images, H the height, W the width and C the number of channels.
            offload_model: Loads the model to CPU after usage.

        Returns:
            List of masks, one for each input frame where an occlusion may exist.

        Raises:
            ValueError if the frame_list has wrong format or if set_parameters was not
                called successfully beforehand.
        """
        batch_size = self.params.get("batch_size", 1)
        depth_threshold = self.params.get("depth_threshold", 0.25)

        img_list: List[Image.Image]
        if isinstance(frame_list, List):
            img_list = [self._prepare_img(img) for img in frame_list]
        else:
            if len(frame_list.shape) < 3:
                raise ValueError(
                    (
                        "Given frame_list as np.ndarray with less than "
                        f"two dimensions: {frame_list.shape}."
                        "\nThe input format of the np.ndarray should be [B, H, W, C]"
                        ", where B is the number of images, H the height, W "
                        "the width and C the number of channels."
                    )
                )
            elif len(frame_list.shape) == 3:
                img_list = [self._prepare_img(frame_list)]
            else:
                img_list = [self._prepare_img(frame_list[i]) for i in range(frame_list.shape[0])]

        ret: List[np.ndarray] = []

        # inference
        pipe_out = self.pipe(img_list, batch_size=batch_size)

        for pipe_res in pipe_out:
            depth_values = pipe_res["predicted_depth"].cpu().numpy()
            depth_values = depth_values - depth_values.min()
            depth_values = depth_values / depth_values.max()

            ret.append(depth_values > depth_threshold)

        if offload_model:
            # TODO figure out how to dynamically offload pipeline
            pass

        return ret

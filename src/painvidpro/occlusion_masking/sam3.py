"""Occlusion Masking with SAM 3."""

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from accelerate import Accelerator
from transformers import Sam3VideoModel, Sam3VideoProcessor

from painvidpro.occlusion_masking.base import OcclusionMaskingBase


class OcclusionMaskingSAM3(OcclusionMaskingBase):
    def __init__(self):
        """Class to detect sequences."""
        super().__init__()
        self._model: Optional[Sam3VideoModel] = None
        self._processor: Optional[Sam3VideoProcessor] = None
        self.set_default_parameters()
        self.device = Accelerator().device
        self.prompt: Union[str, List[str]] = ["a hand"]

    @property
    def model(self) -> Sam3VideoModel:
        if self._model is None:
            raise RuntimeError(
                (
                    "Model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._model

    @property
    def processor(self) -> Sam3VideoProcessor:
        if self._processor is None:
            raise RuntimeError(
                (
                    "Model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._processor

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
        self.prompt = self.params["prompt"]
        try:
            self._model = Sam3VideoModel.from_pretrained(self.params["model"]).to(device, dtype=torch.bfloat16)
            self._processor = Sam3VideoProcessor.from_pretrained(self.params["processor"])
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "device": "cuda",
            "model": "facebook/sam3",
            "processor": "facebook/sam3",
            "convert_input_from_bgr_to_rgb": True,
            "prompt": ["a hand", "a paintbrush", "a pencil"],
        }

    def _prepare_img(self, img: np.ndarray) -> np.ndarray:
        """Converts a numpy array image."""
        convert_input_from_bgr_to_rgb = self.params.get("convert_input_from_bgr_to_rgb", True)
        if convert_input_from_bgr_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def compute_mask_list(
        self, frame_list: Union[List[np.ndarray], np.ndarray], offload_model: bool = True
    ) -> List[np.ndarray]:
        """
        Computes occlusion masks based on the input frames, using the SAM 3 model.

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
        img_list: List[np.ndarray]
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

        # Initialize video inference session
        inference_session = self.processor.init_video_session(
            video=img_list,
            inference_device=self.device,
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.bfloat16,
        )

        # Add text prompt to detect and track objects
        inference_session = self.processor.add_text_prompt(
            inference_session=inference_session,
            text=self.prompt,
        )

        outputs_per_frame = {}
        for model_outputs in self.model.propagate_in_video_iterator(
            inference_session=inference_session, max_frame_num_to_track=None
        ):
            processed_outputs = self.processor.postprocess_outputs(inference_session, model_outputs)
            outputs_per_frame[model_outputs.frame_idx] = processed_outputs

        ret: List[np.ndarray] = []
        for _, item in outputs_per_frame.items():
            masks = item["masks"]
            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy().astype(np.bool)
            # Also works if it did not provide any mask since first dimenstion will be 0
            ret.append(np.any(masks, axis=0))
        return ret

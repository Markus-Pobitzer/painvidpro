"""Occlusion Masking."""

from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageSegmentation

from painvidpro.occlusion_masking.base import OcclusionMaskingBase
from painvidpro.utils.list_processing import batch_list


class OcclusionMaskingRMBG(OcclusionMaskingBase):
    def __init__(self):
        """Class to detect sequences."""
        super().__init__()
        self.processor = None
        self.model = None
        self.set_default_parameters()
        self.image_size = (1024, 1024)
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

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
        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        torch.set_float32_matmul_precision(["high", "highest"][0])
        self.model.eval()
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "model_id": "briaai/RMBG-2.0",
            "device": "cuda",
            "batch_size": 1,
            "prompt": "a hand.",
            "disable_tqdm": True,
        }

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
        device = self.params.get("device", "cuda")
        batch_size = self.params.get("batch_size", 1)
        disable_tqdm = self.params.get("disable_tqdm", True)

        img_array: Union[List[np.ndarray], np.ndarray]
        if isinstance(frame_list, List):
            img_array = frame_list
            numb_imgs = len(img_array)
        else:
            img_array = frame_list
            if len(img_array.shape) < 3:
                raise ValueError(
                    (
                        "Given frame_list as np.ndarray with less than "
                        f"two dimensions: {img_array.shape}."
                        "\nThe input format of the np.ndarray should be [B, H, W, C]"
                        ", where B is the number of images, H the height, W "
                        "the width and C the number of channels."
                    )
                )
            elif len(img_array.shape) == 3:
                # Expanding a dimension
                img_array = np.expand_dims(img_array, axis=0)
            numb_imgs = img_array.shape[0]

        if self.model is None:
            raise ValueError(
                (
                    "Model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        self.model.to(device)

        ret: List[np.ndarray] = []
        image_list: List[torch.Tensor] = []
        shape_list: List[Tuple[int, ...]] = []
        for i in range(numb_imgs):
            shape_list.append(img_array[i].shape)
            image = Image.fromarray(cv2.cvtColor(np.uint8(img_array[i]), cv2.COLOR_BGR2RGB))
            image_list.append(self.transform_image(image).unsqueeze(0))

        index_list = list(range(len(image_list)))
        # Batched inference
        for batch_index_list in tqdm(batch_list(index_list, batch_size=batch_size), disable=disable_tqdm):
            start_idx = batch_index_list[0]
            end_idx = batch_index_list[-1] + 1
            batch_shape_list = shape_list[start_idx:end_idx]
            input_tensor = torch.cat(image_list[start_idx:end_idx], dim=0).to(device)
            # Prediction
            with torch.no_grad():
                preds = self.model(input_tensor)[-1].sigmoid().cpu()

            for i in range(preds.shape[0]):
                height, width, _ = batch_shape_list[i]
                pred = preds[i].squeeze()
                pred_np = pred.cpu().numpy()
                pred_np = cv2.resize(pred_np, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
                pred_np = np.round(pred_np).astype(bool)
                ret.append(pred_np)

        if offload_model:
            self.model.to("cpu")

        return ret

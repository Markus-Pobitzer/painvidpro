# Part of the code from https://github.com/enesmsahin/simple-lama-inpainting/tree/main
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from PIL import Image
from torch.hub import download_url_to_file, get_dir
from tqdm import tqdm

from painvidpro.occlusion_removing.base import OcclusionRemovingBase
from painvidpro.utils.image_processing import process_input


# Source https://github.com/advimman/lama
def get_image(image):
    if isinstance(image, Image.Image):
        img = np.array(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
    else:
        raise Exception("Input image should be either PIL Image or numpy array!")

    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))  # chw
    elif img.ndim == 2:
        img = img[np.newaxis, ...]

    assert img.ndim == 3

    img = img.astype(np.float32) / 255
    return img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(
        img,
        ((0, 0), (0, out_height - height), (0, out_width - width)),
        mode="symmetric",
    )


def prepare_img_and_mask(image, mask, device, pad_out_to_modulo=8, scale_factor=None):
    out_image = get_image(image)
    out_mask = get_image(mask)

    if scale_factor is not None:
        out_image = scale_image(out_image, scale_factor)
        out_mask = scale_image(out_mask, scale_factor, interpolation=cv2.INTER_NEAREST)

    if pad_out_to_modulo is not None and pad_out_to_modulo > 1:
        out_image = pad_img_to_modulo(out_image, pad_out_to_modulo)
        out_mask = pad_img_to_modulo(out_mask, pad_out_to_modulo)

    out_image = torch.from_numpy(out_image).unsqueeze(0).to(device)
    out_mask = torch.from_numpy(out_mask).unsqueeze(0).to(device)

    out_mask = (out_mask > 0) * 1

    return out_image, out_mask


# Source: https://github.com/Sanster/lama-cleaner/blob/6cfc7c30f1d6428c02e21d153048381923498cac/lama_cleaner/helper.py # noqa
def get_cache_path_by_url(url: str) -> str:
    parts = urlparse(url)
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, "checkpoints")
    if not os.path.isdir(model_dir):
        os.makedirs(os.path.join(model_dir, "hub", "checkpoints"))
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    return cached_file


def download_model(url) -> str:
    cached_file = get_cache_path_by_url(url)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        download_url_to_file(url, cached_file, hash_prefix, progress=True)
    return cached_file


LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",  # noqa
)


class SimpleLama:
    def __init__(
        self,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        custom_model_path = os.environ.get("LAMA_MODEL")
        if custom_model_path is not None:
            model_path = custom_model_path
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"lama torchscript model not found: {model_path}")
        else:
            model_path = download_model(LAMA_MODEL_URL)

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.device = device

    def __call__(
        self, image: Union[Image.Image, np.ndarray], mask: Union[Image.Image, np.ndarray], offlaod_model: bool = True
    ) -> np.ndarray:
        self.model.to(self.device)

        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]

        image, mask = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image, mask)

            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            cur_res = cur_res[:height, :width]

        if offlaod_model:
            self.model.to("cpu")
        return cur_res


class OcclusionRemovingLamaInpainting(OcclusionRemovingBase):
    def __init__(self):
        """Class to detect objects."""
        super().__init__()
        self._model: Optional[SimpleLama] = None
        self.set_default_parameters()

    @property
    def model(self) -> SimpleLama:
        if self._model is None:
            raise RuntimeError(
                (
                    "Lama Inpainting model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model first."
                )
            )
        return self._model

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
        try:
            self._model = SimpleLama(device=device)
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "device": "cuda",
            "convert_input_from_bgr_to_rgb": True,
            "disable_tqdm": True,
        }

    def offload_model(self):
        """Offloads the model to CPU, no effect if methdod has no model."""
        self.model.model.to("cpu")

    def remove_occlusions(
        self, frame_list: List[np.ndarray], mask_list: List[np.ndarray], offload_model: bool = True
    ) -> List[np.ndarray]:
        """
        Removes occlusions indicated by the masks.

        Args:
            frame_list: List of frames in cv2 image format. Since cv2 has the BGR
                channeling order it gets automatically converted to a RGB ordering
                if convert_input_from_bgr_to_rgb is specified in the configuration,
                enabled as default.
            mask_list: List of masks in cv2 format.
            offload_model: Offloads the model to CPU after call.

        Returns:
            List of frames where the parts of the frame indicated
            by the masks (occlusions) have been removed. The returned image
            is in RGB format.
        """
        disable_tqdm = self.params.get("disable_tqdm", True)
        convert_input_from_bgr_to_rgb = self.params.get("convert_input_from_bgr_to_rgb", True)

        ret: List[np.ndarray] = []
        for frame, mask in tqdm(zip(frame_list, mask_list), disable=disable_tqdm, desc="Inpainting images"):
            processed_image = process_input(frame, convert_bgr_to_rgb=convert_input_from_bgr_to_rgb)
            processed_mask = process_input(mask, convert_bgr_to_rgb=False)
            ret.append(self.model(processed_image, processed_mask, offlaod_model=False))

        if offload_model:
            self.offload_model()

        return ret

    def remove_occlusions_on_disk(
        self, frame_path_list: List[str], mask_path_list: List[str], output_dir: str, offload_model: bool = True
    ) -> List[str]:
        """
        Removes occlusions indicated by the masks.

        Images are saved to disk and only the paths get returned.

        Args:
            frame_path_list: List of paths to frames representing the video.
            mask_path_list: List of paths to masks corresponding to each frame.
            output_dir: Directory where the output merged frames will be stored.
            offload_model: Offloads the model to CPU after call.

        Returns:
            List of frame paths where the parts of the frame indicated
            by the masks (occlusions) have been removed.
        """
        raise NotImplementedError("Not implemented, use remove_occlusions instead.")

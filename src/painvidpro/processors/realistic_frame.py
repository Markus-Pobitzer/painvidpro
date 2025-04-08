"""Creates a realistic image from a painting using a controlnet."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from controlnet_aux import LineartDetector
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
)
from PIL import Image, ImageOps

from painvidpro.processors.base import ProcessorBase
from painvidpro.utils.image_processing import find_best_aspect_ratio


class ProcessorRealisticFrame(ProcessorBase):
    def __init__(self):
        """Init."""
        super().__init__()
        self.set_default_parameters()
        self._pipe: Optional[StableDiffusionControlNetPipeline] = None
        self.controlnet_map: Dict[str, str] = {"lineart": "lllyasviel/control_v11p_sd15_lineart"}
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
        )
        self.generator = torch.manual_seed(0)
        self.metadata_name = "metadata.json"

    @property
    def pipe(self) -> StableDiffusionControlNetPipeline:
        if self._pipe is None:
            raise RuntimeError(
                (
                    "Diffusion Pipe not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model first."
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
        ret, msg = super().set_parameters(params)
        if not ret:
            return ret, msg

        try:
            controlnet_type = self.params["controlnet_type"]
            if controlnet_type not in self.controlnet_map:
                return False, (
                    f"Value for controlnet_type {controlnet_type} is not supported!\n"
                    f"Choose one of {', '.join(self.controlnet_map.keys())}"
                )
            controlnet_checkpoint = self.controlnet_map[controlnet_type]
            model = self.params["model"]
            device = self.params["device"]
            controlnet = ControlNetModel.from_pretrained(controlnet_checkpoint, torch_dtype=torch.float16)
            self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
            )
            self._pipe.to(device)
            self._pipe.set_progress_bar_config(disable=True)
            self.generator = torch.manual_seed(self.params["seed"])
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params["controlnet_type"] = "lineart"
        self.params["model"] = "emilianJR/epiCRealism"
        self.params["model_size_list"] = [(768, 512), (512, 768)]
        self.params["pad_input"] = False
        self.params["prompt"] = "a real photo"
        self.params["negative_prompt"] = (
            "longbody, lowres, bad anatomy, bad hands, missing fingers,"
            " extra digit, fewer digits, cropped, worst quality, low quality"
        )
        self.params["num_inference_steps"] = 30
        self.params["reference_frame_name"] = "reference_frame.png"
        self.params["device"] = "cuda"
        self.params["seed"] = 123

    def _prepare_image(self, image: Image.Image, inference_size: Tuple[int, int]) -> Image.Image:
        """Prepares the input image by resizing it to the target inference size.

        Args:
            image: Input image to be processed
            inference_size: Target size (width, height) for the inference

        Returns:
            The resized image
        """
        pad_input = self.params.get("pad_input", False)
        if pad_input:
            # Resize image, keep aspect ratio
            image = ImageOps.contain(image, size=inference_size)
            # Pad left side and bottom
            image = ImageOps.pad(image, size=inference_size, color=(255, 255, 255))
        else:
            image = image.resize(inference_size)
        return image

    def _get_control_image(self, image: Image.Image, inference_size: Tuple[int, int]) -> Image.Image:
        """Extracts the control image.

        Args:
            image: The image to create the control image from.
            inference_size: The desired size for inference.

        Returns:
            The control image.
        """
        controlnet_type = self.params["controlnet_type"]
        if controlnet_type == "lineart":
            # Line Art
            processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
            control_image = processor(image)
        else:
            raise RuntimeError(f"Using controlnet_type of {controlnet_type} is not supported!")
        # Ensure that control image is the same size as the desired size of the model
        # The processor may have chenged the size of the image.
        control_image = control_image.resize(inference_size)
        return control_image

    def _post_process_image(self, image: Image.Image, image_size: Tuple[int, int]) -> Image.Image:
        """Post-processes the generated image by resizing it to the original dimensions.

        Args:
            image: Generated image to be processed.
            image_size: Target size (width, height) to resize to.

        Returns:
            The resized image.
        """
        pad_input = self.params.get("pad_input", False)
        if pad_input:
            # Resize image, keep aspect ratio, crop
            image = ImageOps.fit(image, size=image_size)
        else:
            image = image.resize(image_size)
        return image

    def _process(self, video_dir: Path, reference_frame_path: str) -> bool:
        """Processes a single reference frame to generate a realistic version.

        Args:
            video_dir: Directory containing the video frames
            reference_frame_path: Path to the specific reference frame to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        prompt = self.params.get("prompt", "a real photo")
        negative_prompt = self.params.get("negative_prompt", "")
        num_inference_steps = self.params.get("num_inference_steps", 30)
        try:
            image_path = video_dir / reference_frame_path
            image = Image.open(image_path)
        except Exception as e:
            self.logger.info(f"Failed to laod image {str(image_path)} with error: {e}")
            return False
        width, height = image.size
        size_list = self.params.get("model_size_list", [])
        best_resolution = find_best_aspect_ratio(image_size=(width, height), size_list=size_list)
        if best_resolution is None:
            self.logger.info(
                (
                    f"Was not able to select a suitable resolution for image {str(image_path)} in "
                    f"model_size_list {size_list}. Make sure to set model_size_list in the parameters."
                )
            )
            return False
        self.logger.info(f"Selected inference size {best_resolution} for image with size {image.size}.")
        try:
            resized_image = self._prepare_image(image, inference_size=best_resolution)
            control_image = self._get_control_image(resized_image, inference_size=best_resolution)
            final_image = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                generator=self.generator,
                image=control_image,
            ).images[0]
            final_image = self._post_process_image(final_image, image_size=(width, height))
            # Overwriting original Frame
            final_image.save(str(image_path))
            self.logger.info("Successfully replaced original reference frame with generated one.")
        except Exception as e:
            self.logger.info((f"Was not able to generate an image for {str(image_path)}, error: {str(e)}"))
            return False
        return True

    def process(self, video_dir_list: List[str], batch_size: int = -1) -> List[bool]:
        """
        Processes the reference Frames in the video dir.

        The processor creates a realisitc photograph out of the
        drawing/sketch of the reference frame.

        Args:
            video_dir_list: List of paths where the videos are stored.
            batch_size: Gets ignored.

        Returns:
            A list of bools, indidcating for each element in video_dir_list, if the processing
            was successfull.
        """
        ret = [False] * len(video_dir_list)
        for i, vd in enumerate(video_dir_list):
            video_dir = Path(vd)
            log_file = str(video_dir / "ProcessorRealisticFrame.log")
            logging.basicConfig(
                filename=log_file,
                filemode="w",
                force=True,
            )
            reference_frame_name = self.params.get("reference_frame_name", "reference_frame.png")
            reference_frame_path = str(video_dir / reference_frame_name)
            if not self._process(video_dir=video_dir, reference_frame_path=reference_frame_path):
                continue

            ret[i] = True

        # Clear file logging
        logging.basicConfig(
            filename=None,
            force=True,
        )
        return ret

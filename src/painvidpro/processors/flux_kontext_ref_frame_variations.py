"""Creates a variaous style transfered reference frames from a painting using a controlnet."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import FluxKontextPipeline
from PIL import Image
from tqdm import tqdm

from painvidpro.logging.logging import cleanup_logger, setup_logger
from painvidpro.processors.base import ProcessorBase
from painvidpro.utils.image_processing import (
    find_best_aspect_ratio,
    pil_resize_with_padding,
    pil_reverse_resize_with_padding,
)
from painvidpro.utils.list_processing import batch_list
from painvidpro.utils.metadata import load_metadata, save_metadata


class ProcessorFluxKontextRefFrameVariations(ProcessorBase):
    def __init__(self):
        """Init."""
        super().__init__()
        self.set_default_parameters()
        self.reference_frame_name = "reference_frame.png"
        self.metadata_name = "metadata.json"
        self.logger = setup_logger(name=__name__)
        self.zfill_num = 8
        self.generator = torch.manual_seed(0)
        self._flux_kontext_pipe: Optional[FluxKontextPipeline] = None

    @property
    def flux_kontext_pipe(self) -> FluxKontextPipeline:
        if self._flux_kontext_pipe is None:
            raise RuntimeError(
                (
                    "Flux Kontext Pipeline not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model."
                )
            )
        return self._flux_kontext_pipe

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
            self._flux_kontext_pipe = FluxKontextPipeline.from_pretrained(
                self.params["flux_kontext_algorithm"], torch_dtype=torch.bfloat16
            )
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params["flux_kontext_algorithm"] = "black-forest-labs/FLUX.1-Kontext-dev"
        self.params["flux_kontext_config"] = {
            "batch_size": 1,
        }
        self.params["model_size_list"] = [
            (672, 1568),
            (688, 1504),
            (720, 1456),
            (752, 1392),
            (784, 1312),
            (832, 1248),
            (800, 1280),
            (880, 1184),
            (944, 1104),
            (1024, 1024),
            (1104, 944),
            (1184, 880),
            (1280, 800),
            (1248, 832),
            (1312, 784),
            (1392, 752),
            (1456, 720),
            (1504, 688),
            (1568, 672),
        ]
        # Define custom prompts for each art media
        self.params["art_media_to_var_prompt"] = {
            # Convert to pencil sketch with natural graphite lines, cross-hatching, and visible paper texture
            "pencil": {
                # "realistic": "A realistic photograph of this pencil sketch.", # This could be used as augmentation
                "realistic": "Hyper realistic painting",
                "oil": "Oil painting",
            },
            "loomis_pencil": {
                "realistic": "Portrait photograph",
                "acrylic": "Hyper realisitc acrylic painting",
            },
            "oil": {
                "realistic": "Make it real",
                "pencil": "Hyper realistic pencil drawing",
            },
            "acrylic": {
                "realistic": "Hyper realisitc painting",
                "pencil": "A pencil drawing",
            },
        }
        self.params["variations_dir"] = "reference_frame_variations"
        self.params["pad_input"] = False

    def _process(self, video_dir: Path, reference_frame_path: str, disable_tqdm: bool = True) -> bool:
        """Processes a single reference frame to generate several version.

        Args:
            video_dir: Directory containing the video frames
            reference_frame_path: Path to the specific reference frame to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        prompt_dict = self.params.get("art_media_to_var_prompt", {})
        variations_dir = self.params.get("variations_dir", "reference_frame_variations")

        # Loading the metadata dict
        succ, metadata = load_metadata(video_dir, metadata_name=self.metadata_name)
        if not succ:
            self.logger.info(f" Failed opening metadata {str(video_dir / self.metadata_name)}.")
            return False

        if "reference_frame_variations" in metadata:
            self.logger.info(
                (" Metadata already contains variations of reference" " frames, no new frames will be generated.")
            )
            return True

        # Load reference image
        image_path = video_dir / reference_frame_path
        try:
            image = Image.open(image_path)
        except Exception as e:
            self.logger.info(f"Failed to laod image {str(image_path)} with error: {e}")
            return False

        # Estimate best inference resolution
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

        # Generate images
        metadata_entry_list: List[Dict[str, Any]] = []
        try:
            variations_path = video_dir / variations_dir
            variations_path.mkdir(parents=True, exist_ok=True)
            self._flux_kontext_pipe = self.flux_kontext_pipe.to("cuda")
            batch_size = self.params["flux_kontext_config"].get("batch_size", 1)

            if self.params["pad_input"]:
                resized_image = pil_resize_with_padding(image, target_size=best_resolution)
            else:
                resized_image = image.resize(best_resolution)

            media_list = metadata["art_media"]
            media = media_list[0] if len(media_list) > 0 else ""
            media_prompt_dict = prompt_dict.get(media, {})
            if len(media_prompt_dict.keys()) == 0:
                self.logger.info(
                    f"No prompts specified for art media {media}, therefore no reference frame variations will be generated for {str(image_path)}"
                )
                return False

            prompt_keys = []
            prompt_list = []
            for key, p in media_prompt_dict.items():
                prompt_keys.append(key)
                prompt_list.append(p)

            ref_image_list = []
            # Read the sampled frames
            for batched_prompt_list in tqdm(
                batch_list(prompt_list, batch_size=batch_size),
                desc="Generating reference frames",
                disable=disable_tqdm,
            ):
                batch_image_list = [resized_image] * len(batched_prompt_list)
                kontext_image_list = self.flux_kontext_pipe(
                    image=batch_image_list,
                    prompt=batched_prompt_list,
                    guidance_scale=2.5,
                    width=best_resolution[0],
                    height=best_resolution[1],
                ).images
                # Crop and Resize to original image
                if self.params["pad_input"]:
                    ref_image_list += [
                        pil_reverse_resize_with_padding(frame, image.size) for frame in kontext_image_list
                    ]
                else:
                    ref_image_list += [frame.resize(image.size) for frame in kontext_image_list]

            for index, (key_name, prompt, img) in enumerate(zip(prompt_keys, prompt_list, ref_image_list)):
                # Saving frame
                img_name = f"reference_frame_variation_{key_name}_{str(index).zfill(self.zfill_num)}.png"
                out_path = str(variations_path / img_name)
                img.save(out_path)
                metadata_entry_list.append(
                    {
                        "path": str(Path(variations_dir) / img_name),
                        "prompt": prompt,
                        # "negative_prompt": negative_prompt,
                        "model": self.params.get("flux_kontext_algorithm", ""),
                        "processor": self.__class__.__name__,
                    }
                )
            self.logger.info(f"Successfully generated {len(prompt_list)} images for specified frame.")
        except Exception as e:
            self.logger.info((f"Was not able to generate an image for {str(image_path)}, error: {str(e)}"))
            return False

        if len(metadata_entry_list) == 0:
            # This should never happen
            self.logger.info("Was not able to save the metadata of the generated images.")
            return False

        # Save entries in metadata
        metadata["reference_frame_variations"] = metadata.get("reference_frame_variations", []) + metadata_entry_list
        save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)
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
            log_file = str(video_dir / "ProcessorRefFrameVariations.log")
            # Set log file path as name to make logger unique
            self.logger = setup_logger(name=log_file, log_file=log_file)

            reference_frame_name = self.params.get("reference_frame_name", "reference_frame.png")
            reference_frame_path = str(video_dir / reference_frame_name)

            if not self._process(video_dir=video_dir, reference_frame_path=reference_frame_path):
                continue

            ret[i] = True

        # Clear file logging
        cleanup_logger(self.logger)
        return ret

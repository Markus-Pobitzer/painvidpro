"""Creates a variaous style transfered reference frames from a painting using a controlnet."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from painvidpro.image_tagger.wd_tagger import Predictor
from painvidpro.processors.realistic_frame import ProcessorRealisticFrame
from painvidpro.utils.image_processing import find_best_aspect_ratio
from painvidpro.utils.metadata import load_metadata, save_metadata


class ProcessorRefFrameVariations(ProcessorRealisticFrame):
    def __init__(self):
        """Init."""
        super().__init__()
        self.set_default_parameters()
        self.logger = logging.getLogger(__name__)
        self.zfill_num = 8
        self._image_tagger: Optional[Predictor] = None

    @property
    def image_tagger(self) -> Predictor:
        if self._image_tagger is None:
            raise RuntimeError(
                (
                    "Iamge Tagger not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model first."
                )
            )
        return self._image_tagger

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        ret, msg = super().set_parameters(params)
        try:
            if self.params["extend_prompt_with_tags"]:
                self._image_tagger = Predictor()
                self.image_tagger.load_model()
        except Exception as e:
            return False, f"An error occurred when loading the Image Tagger: {e}"
        return ret, msg

    def set_default_parameters(self):
        super().set_default_parameters()
        self.params["art_media_to_var_prompt"] = {
            "default": {
                "real": "a real photo",
                "oil": "an oil painting, masterpiece",
                "acrylic": "a realistic acrylic painting, masterpiece",
                "pencil": "a pencil drawing",
            },
            "oil": {
                "pencil": "pencil drawing with natural graphite lines, cross-hatching, and visible paper texture",
            },
            "acrylic": {
                "pencil": "pencil drawing with natural graphite lines, cross-hatching, and visible paper texture",
            },
        }
        self.params["variations_dir"] = "reference_frame_variations"
        self.params["pad_input"] = True
        self.params["extend_prompt_with_tags"] = True

    def _process(self, video_dir: Path, reference_frame_path: str) -> bool:
        """Processes a single reference frame to generate several version.

        Args:
            video_dir: Directory containing the video frames
            reference_frame_path: Path to the specific reference frame to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        variations_dir = self.params.get("variations_dir", "reference_frame_variations")
        negative_prompt = self.params.get("negative_prompt", "")
        num_inference_steps = self.params.get("num_inference_steps", 30)

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

        # Get the prompt list
        prompt_dict = self.params.get("art_media_to_var_prompt", {})
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

        if len(prompt_list) == 0:
            self.logger.info("Prompt list is empty, no images were generated.")
            return False

        # Extend the prompt
        if self.params["extend_prompt_with_tags"]:
            sorted_general_strings, _, _, _ = self.image_tagger.predict(image)
            tags = [
                tag.strip()
                for tag in sorted_general_strings.split(",")
                if tag not in ["no humans"] and "media" not in tag and "medium" not in tag
            ]
            prompt_add = ", ".join(tags[:5])
            if prompt_add != "":
                prompt_list = [prompt + ", " + prompt_add for prompt in prompt_list]

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
            resized_image = self._prepare_image(image, inference_size=best_resolution)
            control_image = self._get_control_image(resized_image, inference_size=best_resolution)
            variations_path = video_dir / variations_dir
            variations_path.mkdir(parents=True, exist_ok=True)

            for index, prompt in enumerate(prompt_list):
                key_name = prompt_keys[index]
                # TODO: batch inference not supported yet
                final_image = self.pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    generator=self.generator,
                    image=control_image,
                ).images[0]
                final_image = self._post_process_image(final_image, image_size=(width, height))
                # Saving frame
                final_frame_name = f"reference_frame_variation_{key_name}_{str(index).zfill(self.zfill_num)}.png"
                out_path = str(variations_path / final_frame_name)
                final_image.save(out_path)
                metadata_entry_list.append(
                    {
                        "path": str(Path(variations_dir) / final_frame_name),
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "num_inference_steps": num_inference_steps,
                        "model": self.params.get("model", ""),
                        "controlnet_type": self.params.get("controlnet_type", ""),
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

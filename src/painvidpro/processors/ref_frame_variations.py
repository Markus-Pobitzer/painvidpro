"""Creates a variaous style transfered reference frames from a painting using a controlnet."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

from painvidpro.processors.realistic_frame import ProcessorRealisticFrame
from painvidpro.utils.image_processing import find_best_aspect_ratio


class ProcessorRefFrameVariations(ProcessorRealisticFrame):
    def __init__(self):
        """Init."""
        super().__init__()
        self.set_default_parameters()
        self.logger = logging.getLogger(__name__)
        self.zfill_num = 8

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        ret, msg = super().set_parameters(params)
        return ret, msg

    def set_default_parameters(self):
        super().set_default_parameters()
        self.params["prompt_list"] = [
            "a real photo",
            "an oil painting, masterpiece",
            "a realistic acrylic painting, masterpiece",
            "a pencil drawing",
        ]
        self.params["variations_dir"] = "reference_frame_variations"

    def _process(self, video_dir: Path, reference_frame_path: str) -> bool:
        """Processes a single reference frame to generate several version.

        Args:
            video_dir: Directory containing the video frames
            reference_frame_path: Path to the specific reference frame to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        prompt_list = self.params.get("prompt_list", [])
        variations_dir = self.params.get("variations_dir", "reference_frame_variations")
        if len(prompt_list) == 0:
            self.logger.info("Prompt list is empty, no images were generated.")
            return False

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
            variations_path = video_dir / variations_dir
            variations_path.mkdir(parents=True, exist_ok=True)

            for index, prompt in enumerate(prompt_list):
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
                final_frame_name = f"reference_frame_variation_{str(index).zfill(self.zfill_num)}.png"
                out_path = str(variations_path / final_frame_name)
                final_image.save(out_path)
            self.logger.info(f"Successfully generated {len(prompt_list)} images for specified frame.")
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

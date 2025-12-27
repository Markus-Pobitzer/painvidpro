"""Creates a realistic image from a painting using a controlnet."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from painvidpro.data_storage.hdf5_video_archive import DynamicVideoArchive
from painvidpro.image_tagger.base import ImageTaggerBase
from painvidpro.image_tagger.llava_next import ImageTaggerLLaVANeXT
from painvidpro.logging.logging import cleanup_logger, setup_logger
from painvidpro.processors.base import ProcessorBase


class ProcessorRefFrameTagging(ProcessorBase):
    def __init__(self):
        """Init."""
        super().__init__()
        self.set_default_parameters()
        self._tagger: Optional[ImageTaggerBase] = None
        self.logger = setup_logger(name=__name__)
        self.frame_data = "frame_data.h5"

    @property
    def tagger(self) -> ImageTaggerBase:
        if self._tagger is None:
            raise RuntimeError(
                (
                    "Image Tagger not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model first."
                )
            )
        return self._tagger

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
            if self.params["image_tagger"] != "ImageTaggerLLaVANeXT":
                # TODO: Create Iamge Tagging Factory and support more options
                return False, "The only supported image_tagger is ImageTaggerLLaVANeXT"
            self._tagger = ImageTaggerLLaVANeXT()
            succ, msg = self._tagger.set_parameters(self.params["image_tagger_config"])
            if not succ:
                return False, msg
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params["image_tagger"] = "ImageTaggerLLaVANeXT"
        self.params["image_tagger_config"] = {"device": "cuda", "torch_dtype": "int4"}
        self.params["reference_frame_name"] = "reference_frame.png"
        prompt_dict = {
            "acrylic": """Describe the visual content of the image as if you were guiding someone to recreate the scene with an image generation model, in a single sentence or two. Focus only on the objects and describe their colors. Do not mention anything about the artistic style, medium, or that it is a drawing or sketch.""",
            "loomis_pencil": """"Describe the visual content of the image as if you were guiding someone to recreate the scene with an image generation model, in a single sentence or two. Focus only on the objects, people, environment, and their relationships. Do not mention anything about the artistic style, medium, color, or that it is a drawing or sketch.""",
            "oil": """Describe the visual content of the image as if you were guiding someone to recreate the scene with an image generation model, in a single sentence or two. Focus only on the objects and describe their colors. Do not mention anything about the artistic style, medium, or that it is a drawing or sketch.""",
            "pencil": """"Describe the visual content of the image as if you were guiding someone to recreate the scene with an image generation model, in a single sentence or two. Focus only on the objects, people, environment, and their relationships. Do not mention anything about the artistic style, medium, color, or that it is a drawing or sketch.""",
        }
        self.params["art_media_to_tag_prompt"] = prompt_dict

    def _process(self, frame_data: DynamicVideoArchive) -> bool:
        """Processes a single reference frame to generate a description/tag for it.

        Args:
            frame_data: The DynamicVideoArchive.
            reference_frame_path: Path to the specific reference frame to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Loading the metadata dict
        with frame_data:
            metadata = frame_data.get_global_metadata()
            if frame_data.len_reference_frames() == 0:
                self.logger.info("No reference frame found in Video Archive, skipping the tagging process.")
                return False
            image = frame_data.get_reference_frame(0)

        if "reference_frame_tags" in metadata:
            if metadata["reference_frame_tags"] is None:
                # This should not happen ...
                metadata["reference_frame_tags"] = []

            for tag in metadata["reference_frame_tags"]:  # type: ignore
                if tag["image_tagger"] == self.params["image_tagger"]:
                    self.logger.info(
                        (
                            "Metadata already contains image tag for specified image tagger, no new tag will be generated."
                        )
                    )
                    return True

        # Get the prompt
        prompt_dict = self.params.get("art_media_to_tag_prompt", {})
        media_list = metadata.get("art_media", [])
        media = media_list[0] if len(media_list) > 0 else ""
        if media == "pencil" and "loomis" in metadata.get("art_style", []):
            media = "loomis_pencil"
        prompt = prompt_dict.get(media)

        # Generate tags for images
        tag_entry: Dict[str, Any] = {}
        try:
            tag_dict = self.tagger.predict(image=image, prompt=prompt)
            tag = tag_dict["image_description"]
            tag_entry = {
                "prompt": prompt,
                "image_tagger": self.params.get("image_tagger", ""),
                "processor": self.__class__.__name__,
                "tag": tag,
            }
            self.logger.info(f"Successfully generated following tag for the reference frame:\n{tag}")
        except Exception as e:
            self.logger.info((f"Was not able to generate ref frame tags, error: {str(e)}"))
            return False

        # Save entries in metadata
        with frame_data:
            metadata = frame_data.get_global_metadata()
            frame_data.set_global_metadata(
                "reference_frame_tags", metadata.get("reference_frame_tags", []) + [tag_entry]
            )
        return True

    def process(self, video_dir_list: List[str], batch_size: int = -1) -> List[bool]:
        """
        Creates a description/tag for the reference Frame in the video dir.

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
            log_file = str(video_dir / "ProcessorRefFrameTagging.log")
            self.logger = setup_logger(name=log_file, log_file=log_file)
            frame_dataset = DynamicVideoArchive(str(video_dir / self.frame_data))

            if not self._process(frame_data=frame_dataset):
                continue

            ret[i] = True

        # Clear file logging
        cleanup_logger(self.logger)
        return ret

"""Creates a variaous style transfered reference frames from a painting using a controlnet."""

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline, QwenImageEditPlusPipeline
from tqdm import tqdm

from painvidpro.data_storage.hdf5_video_archive import DynamicVideoArchive
from painvidpro.logging.logging import cleanup_logger, setup_logger
from painvidpro.processors.base import ProcessorBase
from painvidpro.utils.list_processing import batch_list


class ProcessorQwenEditRefFrameVariations(ProcessorBase):
    def __init__(self):
        """Init."""
        super().__init__()
        self.set_default_parameters()
        self.reference_frame_name = "reference_frame.png"
        self.metadata_name = "metadata.json"
        self.logger = setup_logger(name=__name__)
        self._pipe: Optional[Union[QwenImageEditPipeline, QwenImageEditPlusPipeline]] = None
        self.torch_dtype = torch.bfloat16
        self.seed: Optional[int] = None
        self.device = "cuda"
        self.frame_data = "frame_data.h5"
        # Set up the generator for reproducibility
        self.generator = torch.Generator(device=self.device).manual_seed(42)
        # From
        # https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
        self.scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }

    @property
    def pipe(self) -> Union[QwenImageEditPipeline, QwenImageEditPlusPipeline]:
        if self._pipe is None:
            raise RuntimeError(
                (
                    "Qwen Image Edit Pipeline not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model."
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
            self.device = self.params["device"]
            self.seed = self.params["seed"]
            # Initialize scheduler with Lightning config
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler_config)
            # Load the edit pipeline with Lightning scheduler
            if self.params["qwen_algorithm"] == "Qwen/Qwen-Image-Edit-2509":
                self._pipe = QwenImageEditPlusPipeline.from_pretrained(
                    self.params["qwen_algorithm"], scheduler=scheduler, torch_dtype=self.torch_dtype
                )
            else:
                self._pipe = QwenImageEditPipeline.from_pretrained(
                    self.params["qwen_algorithm"], scheduler=scheduler, torch_dtype=self.torch_dtype
                )
            self._pipe.load_lora_weights(
                self.params["qwen_lightning_lora"], weight_name=self.params["qwen_lightning_lora_weight_name"]
            )
            self._pipe.fuse_lora()

            if self.params["enable_sequential_cpu_offload"]:
                self._pipe.enable_sequential_cpu_offload()
            else:
                self._pipe.to(self.device)
            # Set up the generator for reproducibility
            if self.seed is None:
                self.seed = random.randint(0, 1024**3)
            self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        # Default Qwen Image Edit
        # self.params["qwen_algorithm"] = "Qwen/Qwen-Image-Edit"
        # self.params["qwen_config"] = {"batch_size": 1}
        # self.params["qwen_lightning_lora"] = "lightx2v/Qwen-Image-Lightning"
        # self.params["qwen_lightning_lora_weight_name"] = "Qwen-Image-Lightning-8steps-V1.1.safetensors"
        # Version 2509
        self.params["qwen_algorithm"] = "Qwen/Qwen-Image-Edit-2509"
        self.params["qwen_config"] = {"batch_size": 1}
        self.params["qwen_lightning_lora"] = "lightx2v/Qwen-Image-Lightning"
        self.params["qwen_lightning_lora_weight_name"] = (
            "Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
        )

        self.params["enable_sequential_cpu_offload"] = False
        self.params["device"] = "cuda"
        self.params["seed"] = 123456

        # Define custom prompts for each art media
        self.params["art_media_to_var_prompt"] = {
            "pencil": {
                "realistic": "Convert to a realistic photo",
                "painting": "Convert the pencil drawing into a realistic painting using natural, true-to-life colors.",
                "monochrome": "To pencil painting, monochrome",
                "color_pencil": "To childish pencil drawing",
                "coloring_book": "To coloring book, monochrome",
            },
            "colored pencils": {
                "realistic": "Convert to a realistic photo",
                "monochrome": "To pencil painting, monochrome",
                "color_pencil": "To childish pencil drawing",
                "coloring_book": "To coloring book, monochrome",
            },
            "loomis_pencil": {
                "realistic": "Render as a high-quality portrait photograph with natural lighting, sharp focus on the subject's face, soft background blur, and realistic skin tones.",
                "painting": "Render as a hyper-realistic painting with fine detail, lifelike textures, accurate lighting, and natural color tones.",
            },
            "oil": {
                "monochrome": "To pencil painting, monochrome",
                "color_pencil": "To childish pencil drawing",
                "coloring_book": "To coloring book, monochrome",
                # "charcol": "To charcol sketch, monochrome", # Very similar to pencil painting monochrome
                # "pencil": "Convert the image to a black-and-white pencil sketch with clear outlines and realistic shading.",
                # "paintlane": "Convert the image into a simple monochrome pencil drawing in the style of PAINTLANE: clean outlines, minimal shading, no color, beginner-friendly, clear and recognizable forms.",
                # "outlines": "Convert the image to a black-and-white pencil sketch showing only the outlines of objects with simple, light shading.",
            },
            "acrylic": {
                "monochrome": "To pencil painting, monochrome",
                "color_pencil": "To childish pencil drawing",
                "coloring_book": "To coloring book, monochrome",
                # "charcol": "To charcol sketch, monochrome", # Very similar to pencil painting monochrome
                # "pencil": "Convert the image to a black-and-white pencil sketch with clear outlines and realistic shading.",
                # "paintlane": "Convert the image into a simple monochrome pencil drawing in the style of PAINTLANE: clean outlines, minimal shading, no color, beginner-friendly, clear and recognizable forms.",
                # "outlines": "Convert the image to a black-and-white pencil sketch showing only the outlines of objects with simple, light shading.",
            },
        }
        self.params["variations_dir"] = "reference_frame_variations"
        self.params["pad_input"] = False

    def _process(self, frame_data: DynamicVideoArchive, disable_tqdm: bool = True) -> bool:
        """Processes a single reference frame to generate several version.

        Args:
            frame_data: The DynamicVideoArchive.
            reference_frame_path: Path to the specific reference frame to process.

        Returns:
            bool: True if processing was successful, False otherwise.
        """
        prompt_dict = self.params.get("art_media_to_var_prompt", {})

        # Keeps track for which prompts we already have reference frame variations
        covered_prompts: Set = set()

        with frame_data:
            num_ref_frames = frame_data.len_reference_frames()
            if num_ref_frames == 0:
                self.logger.info((" No reference frame was found, therefore no variations will be generated."))
                return False
            elif num_ref_frames > 1:
                # The case where variations already exist
                for idx in range(1, num_ref_frames):
                    metadata = frame_data.get_reference_frame_metadata(idx)
                    if str(metadata.get("peocessor", "")) == str(__name__):
                        covered_prompts.add(str(metadata.get("prompt", "")))
            # Index 0 is original referene frame
            image = frame_data.get_reference_frame(index=0)

        # Generate images
        try:
            batch_size = self.params["qwen_config"].get("batch_size", 1)

            # Get the prompt list
            media_list = metadata.get("art_media", [])
            media = media_list[0] if len(media_list) > 0 else ""
            if media == "pencil" and "loomis" in metadata.get("art_style", []):
                media = "loomis_pencil"
            media_prompt_dict = prompt_dict.get(media, {})
            if len(media_prompt_dict.keys()) == 0:
                self.logger.info(
                    f"No prompts specified for art media {media}, therefore no reference frame variations will be generated."
                )
                return False

            prompt_keys = []
            prompt_list = []
            for key, p in media_prompt_dict.items():
                if p in covered_prompts:
                    self.logger.info(f"Skipping prompt {p} since it is already contained.")

                prompt_keys.append(key)
                prompt_list.append(p)

            ref_image_list = []

            # Hardcode the negative prompt as in the original
            negative_prompt = " "

            seed_list = []
            # We rsize to 1024 since this seems to be the best resoultion
            # to not get a zoom in or out. However it changes aspect ratio ...
            in_img = image.resize((1024, 1024))
            # Read the sampled frames
            for batched_prompt_list in tqdm(
                batch_list(prompt_list, batch_size=batch_size),
                desc="Generating reference frames",
                disable=disable_tqdm,
            ):
                batch_image_list = [in_img] * len(batched_prompt_list)
                negative_prompt_list = [negative_prompt] * len(batched_prompt_list)
                # I think if we generate more than one image this may be off
                seed_list += [self.generator.seed()] * len(batched_prompt_list)

                with torch.inference_mode():
                    kontext_image_list = self.pipe(
                        batch_image_list,
                        prompt=batched_prompt_list,
                        negative_prompt=negative_prompt_list,
                        num_inference_steps=8,
                        generator=self.generator,
                        true_cfg_scale=1.0,
                        num_images_per_prompt=1,  # Always generate only 1 image
                    ).images

                # Resizing to original size
                ref_image_list += [frame.resize(image.size) for frame in kontext_image_list]

            with frame_data:
                for index, (prompt, img) in enumerate(zip(prompt_list, ref_image_list)):
                    metadata_dit = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": seed_list[index],
                        "model": self.params.get("qwen_algorithm", ""),
                        "lora": self.params.get("qwen_lightning_lora_weight_name", ""),
                        "processor": self.__class__.__name__,
                    }
                    frame_data.add_reference_frame(image=img, metadata_dict=metadata_dit)
            self.logger.info(f"Successfully generated {len(prompt_list)} images for specified frame.")
        except Exception as e:
            self.logger.info((f"Was not able to generate reference frame variations, error: {str(e)}"))
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
            log_file = str(video_dir / "ProcessorQwenEditRefFrameVariations.log")
            # Set log file path as name to make logger unique
            self.logger = setup_logger(name=log_file, log_file=log_file)
            frame_dataset = DynamicVideoArchive(str(video_dir / self.frame_data))

            if not self._process(frame_data=frame_dataset):
                continue

            ret[i] = True

        # Clear file logging
        cleanup_logger(self.logger)
        return ret

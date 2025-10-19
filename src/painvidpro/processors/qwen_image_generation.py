"""Creates a variaous style transfered reference frames from a painting using a controlnet."""

import math
import random
from os.path import isfile, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImagePipeline

from painvidpro.logging.logging import cleanup_logger, setup_logger
from painvidpro.processors.base import ProcessorBase
from painvidpro.utils.metadata import load_metadata, save_metadata


class ProcessorQwenImageGeneration(ProcessorBase):
    def __init__(self):
        """Init."""
        super().__init__()
        self.set_default_parameters()
        self.reference_frame_name = "reference_frame.png"
        self.metadata_name = "metadata.json"
        self.extr_folder_name = "extracted_frames"
        self.logger = setup_logger(name=__name__)
        self.zfill_num = 8
        self._pipe: Optional[QwenImagePipeline] = None
        self.torch_dtype = torch.bfloat16
        self.seed: Optional[int] = None
        self.device = "cuda"
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
    def pipe(self) -> QwenImagePipeline:
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
            self._pipe = QwenImagePipeline.from_pretrained(
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
        self.params["qwen_algorithm"] = "Qwen/Qwen-Image"
        self.params["qwen_config"] = {"batch_size": 1}
        self.params["qwen_lightning_lora"] = "lightx2v/Qwen-Image-Lightning"
        self.params["qwen_lightning_lora_weight_name"] = (
            "Qwen-Image-Lightning/Qwen-Image-Lightning-8steps-V2.0.safetensors"
        )

        self.params["enable_sequential_cpu_offload"] = False
        self.params["device"] = "cuda"
        self.params["seed"] = 123456

    def _process(self, video_dir: Path, reference_frame_path: str, disable_tqdm: bool = True) -> bool:
        """Generates the refernece frame.

        Args:
            video_dir: Directory containing the video frames
            reference_frame_path: Path to the specific reference frame to process

        Returns:
            bool: True if processing was successful, False otherwise
        """
        if isfile(str(reference_frame_path)):
            return True

        # Loading the metadata dict
        succ, metadata = load_metadata(video_dir, metadata_name=self.metadata_name)
        if not succ:
            self.logger.info(f" Failed opening metadata {str(video_dir / self.metadata_name)}.")
            return False

        try:
            prompt = ""
            if "reference_frame_tags" in metadata and len(metadata["reference_frame_tags"]) > 0:
                prompt = metadata["reference_frame_tags"][0]["tag"]
            elif "titel" in metadata:
                prompt = metadata["titel"]
            else:
                self.logger.info(
                    "Was not able to determine which prompt to use. No reference_frame_tags nor titel specified in metadata."
                )
                return False
            # Hardcode the negative prompt as in the original
            negative_prompt = " "
            seed = self.generator.seed()
            with torch.inference_mode():
                ref_frame = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=8,
                    generator=self.generator,
                    true_cfg_scale=1.0,
                    num_images_per_prompt=1,  # Always generate only 1 image
                ).images[0]

            ref_frame.save(reference_frame_path)
            # Hard code these values since they have no meaning here
            max_index = 100
            metadata["start_frame_idx"] = 0
            metadata["end_frame_idx"] = max_index
            # Save it as the last frame in the video
            extr_frame_dir = video_dir / self.extr_folder_name
            extr_frame_dir.mkdir(parents=True, exist_ok=True)
            frame_name = f"frame_{str(max_index).zfill(self.zfill_num)}.png"
            full_frame_path = str(extr_frame_dir / frame_name)
            ref_frame.save(full_frame_path)
            rel_frame_path = join(self.extr_folder_name, frame_name)
            metadata["extracted_frames"] = [
                {
                    "index": max_index,
                    "path": rel_frame_path,
                    "extraction_method": "generated",
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "seed": seed,
                    "model": self.params.get("qwen_algorithm", ""),
                    "lora": self.params.get("qwen_lightning_lora_weight_name", ""),
                    "processor": self.__class__.__name__,
                }
            ]

            # Save entries in metadata
            save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)

            self.logger.info("Successfully generated reference frame.")
        except Exception as e:
            self.logger.info((f"Was not able to generate an image for {str(video_dir)}, error: {str(e)}"))
            return False

        return True

    def process(self, video_dir_list: List[str], batch_size: int = -1) -> List[bool]:
        """
        Generates the reference Frame in the video dir.

        The processor generates an image based on the provided prompt or image tag.
        The reference frame also gets taken as the last frame in the video, no other
        frames for the painting process get generated.

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
            log_file = str(video_dir / "ProcessorQwenImageGeneration.log")
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

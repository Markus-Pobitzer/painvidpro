from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaNextForConditionalGeneration

from painvidpro.image_tagger.base import ImageTaggerBase


class ImageTaggerLLaVANeXT(ImageTaggerBase):
    def __init__(self):
        """Class to describe image."""
        self._processor: Optional[Any] = None
        self._model: Optional[LlavaNextForConditionalGeneration] = None
        self.set_default_parameters()

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError(
                (
                    "Model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._processor

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError(
                (
                    "Model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
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

        self.device = self.params["device"]
        try:
            self.model_name = self.params["model"]
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            if self.params["torch_dtype"] == "int4":
                # Quantization
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
                )
                self._model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name, quantization_config=quant_config, device_map="auto"
                )
            else:
                torch_dtype = torch.float16
                if self.params["torch_dtype"] == "float32":
                    torch_dtype = torch.float32
                elif self.params["torch_dtype"] == "bfloat16":
                    torch_dtype = torch.bfloat16

                self._model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_name, torch_dtype=torch_dtype
                ).to(self.device)
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "device": "cuda",
            "torch_dtype": "float16",
            "model": "llava-hf/llava-v1.6-mistral-7b-hf",
            "default_prompt": """"Describe the visual content of the image as if you were guiding someone to recreate the scene with an image generation model, in a single sentence or two. Focus only on the objects, people, environment, and their relationships. Do not mention anything about the artistic style, medium, color, or that it is a drawing or sketch.""",
        }

    def predict(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if prompt is None:
            prompt = self.params["default_prompt"]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        vlm_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(image, vlm_prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        output = self.model.generate(**inputs, max_new_tokens=100)
        img_desc = self.processor.decode(output[0][input_length:], skip_special_tokens=True)
        return {"image_description": img_desc}

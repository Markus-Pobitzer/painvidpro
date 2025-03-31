"""Object detection with grounding dino."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from painvidpro.object_detection.base import ObjectDetectionBase
from painvidpro.utils.image_processing import process_input
from painvidpro.utils.list_processing import batch_list


class ObjectDetectionGroundingDino(ObjectDetectionBase):
    def __init__(self):
        """Class to detect objects."""
        super().__init__()
        self._processor: Optional[Any] = None
        self._model: Optional[Any] = None
        self.set_default_parameters()

    @property
    def processor(self) -> Any:
        if self._processor is None:
            raise RuntimeError(
                (
                    "Object Detection model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._processor

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError(
                (
                    "Object Detection model not correctly instanciated. Make sure to call "
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

        model_id = self.params["model_id"]
        try:
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "model_id": "IDEA-Research/grounding-dino-tiny",
            "device": "cuda",
            "batch_size": 1,
            "convert_input_from_bgr_to_rgb": True,
            "prompt": "a hand.",
            "box_threshold": 0.6,
            "text_threshold": 0.6,
            "disable_tqdm": True,
        }

    def offload_model(self):
        """Offloads the model to CPU, no effect if methdod has no model."""
        self.model.to("cpu")

    def _object_detection(self, image_list: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """Detects objects with Grounding Dino.

        Args:
            image_list: List of images.

        Returns:
            A List with an entry for each image in image_list. The entry consists
            of a List of Dicts containig the label and sores.
        """
        text = self.params.get("prompt", "a hand.")
        box_threshold = self.params.get("box_threshold", 0.6)
        text_threshold = self.params.get("text_threshold", 0.6)
        device = self.params.get("device", "cuda")
        inputs = self.processor(images=image_list, text=[text] * len(image_list), return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        height, width, _ = image_list[0].shape
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)] * len(image_list),
        )

        ret: List[List[Dict[str, Any]]] = []
        for res in results:
            frame_ret: List[Dict[str, Any]] = []
            scores = res["scores"].cpu().detach().numpy()
            boxes = res["boxes"].cpu().detach().numpy()
            for i, label in enumerate(res["labels"]):
                if label == "":
                    continue
                frame_ret.append(
                    {
                        "label": label,
                        "score": float(scores[i]),
                        "box": boxes[i],
                    }
                )

            ret.append(frame_ret)
        return ret

    def detect_objects(
        self, frame_list: Union[List[np.ndarray], List[str]], offload_model: bool = True
    ) -> List[List[Dict[str, Any]]]:
        """Detects objets from frames with Grounding Dino.

        Returns a List of detected objects, indicating for each frame if the object is present.
        The desired objects should be set as parameters.

        Args:
            frame_list: List of frames in cv2 image format or paths. Since cv2 has the BGR
                channeling order it gets automatically converted to a RGB ordering
                if convert_input_from_bgr_to_rgb is specified in the configuration,
                enabled as default.
            offload_model: Loads the model to CPU after usage.

        Returns:
            List of Lists indicatinig detected objects. For each image a List of dicts.
        """
        device = self.params.get("device", "cuda")
        batch_size = self.params.get("batch_size", 1)
        disable_tqdm = self.params.get("disable_tqdm", True)
        convert_input_from_bgr_to_rgb = self.params.get("convert_input_from_bgr_to_rgb", True)
        self.model.to(device)

        ret: List[List[Dict[str, Any]]] = []
        for batch in tqdm(batch_list(frame_list, batch_size), disable=disable_tqdm, desc="Detecting objects"):
            batch_img_list = [
                process_input(batch_frame, convert_bgr_to_rgb=convert_input_from_bgr_to_rgb) for batch_frame in batch
            ]
            ret += self._object_detection(batch_img_list)

        if offload_model:
            self.model.to("cpu")

        return ret

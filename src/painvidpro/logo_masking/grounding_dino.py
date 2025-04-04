"""Object detection with grounding dino."""

from typing import Any, Dict, List, Tuple

import numpy as np

from painvidpro.logo_masking.base import LogoMaskingBase
from painvidpro.object_detection.grounding_dino import ObjectDetectionGroundingDino


class LogoMaskingGroundingDino(LogoMaskingBase, ObjectDetectionGroundingDino):
    def __init__(self):
        """Class to detect objects."""
        ObjectDetectionGroundingDino.__init__(self)

    def set_default_parameters(self):
        ObjectDetectionGroundingDino.set_default_parameters(self)
        self.params["prompt"] = "a watermark. a logo. a text."
        # Reduce the thresholds
        self.params["box_threshold"] = 0.4
        self.params["text_threshold"] = 0.4
        self.params["bbox_max_area"] = 0.25

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        return ObjectDetectionGroundingDino.set_parameters(self, params=params)

    def offload_model(self):
        """Offloads the model to CPU, no effect if methdod has no model."""
        return ObjectDetectionGroundingDino.offload_model(self)

    def bbox_area_too_large(self, image: np.ndarray, bbox: Tuple[int, int, int, int], threshold: float = 0.25):
        """Checks if the realtve bounding box area is bigger than the threshold."""
        rel_width = bbox[2] - bbox[0]
        rel_height = bbox[3] - bbox[1]
        area = rel_width * rel_height
        img_area = image.shape[1] * image.shape[0]
        rel_area = area / img_area
        return rel_area > threshold

    def _create_detection_mask(self, image: np.ndarray, detection_list: List[Dict[str, Any]]):
        """Creates binary mask where objects were detected.

        Args:
            image: The image.
            detection_list: The list of detected objects.

        Returns:
            An array with a binary mask indidcating where the object is.
        """
        mask = np.zeros(image.shape[:2], dtype=bool)
        threshold = self.params.get("bbox_max_area", 0.25)

        for detection in detection_list:
            box = detection["box"]
            # Extract and round coordinates to nearest integer
            x_min, y_min, x_max, y_max = np.round(box).astype(int)

            # Clamp coordinates to ensure they are within image bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)  # image width (columns)
            y_max = min(image.shape[0], y_max)  # image height (rows)

            # Skip invalid boxes (where min >= max after clamping)
            if x_min >= x_max or y_min >= y_max:
                continue

            if self.bbox_area_too_large(image=image, bbox=(x_min, y_min, x_max, y_max), threshold=threshold):
                continue

            # Set the region in the mask to True
            mask[y_min:y_max, x_min:x_max] = True

        return mask

    def compute_mask_list(self, frame_list: List[np.ndarray], offload_model: bool = True) -> List[np.ndarray]:
        """Logo mask indicating Text.

        All frames get analyzed and checked which areas contain text.

        Args:
            frame_list: List of frames in cv2 image format.
            offload_model: Gets ignored.

        Returns:
            List with a binary mask for each image.
        """
        ret: List[np.ndarray] = []
        result_list = self.detect_objects(frame_list=frame_list, offload_model=offload_model)
        for img, detection_list in zip(frame_list, result_list):
            ret.append(self._create_detection_mask(img, detection_list))

        return ret

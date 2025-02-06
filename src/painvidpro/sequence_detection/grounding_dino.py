"""Fixed frame number for the relevant sequence detection."""

from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from painvidpro.sequence_detection.base import BaseSequence, SequenceDetectionBase
from painvidpro.utils.image_processing import process_input
from painvidpro.utils.list_processing import batch_list
from painvidpro.video_processing.utils import video_capture_context


class SequenceDetectionGroundingDino(SequenceDetectionBase):
    def __init__(self):
        """Class to detect sequences."""
        super().__init__()
        self.processor = None
        self.model = None
        self.set_default_parameters()

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
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "model_id": "IDEA-Research/grounding-dino-tiny",
            "device": "cuda",
            "batch_size": 1,
            "prompt": "a hand.",
            "box_threshold": 0.4,
            "text_threshold": 0.3,
            "max_frames_to_consider": 200,
            "frame_steps": 10,
        }

    def _object_detection(self, image_list: List[np.ndarray]) -> Tuple[List[List[str]], List[np.ndarray]]:
        """Detects objects with Grounding Dino.

        Args:
            image_list: List of images.

        Returns:
            Tuple with the label_list and associated score_list, where each entry is associated to an image in image_list.
        """
        text = self.params.get("prompt", "a hand")
        box_threshold = self.params.get("box_threshold", 0.4)
        text_threshold = self.params.get("text_threshold", 0.3)
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
        label_list = [res["labels"] for res in results]
        scores_list = [res["scores"].cpu().detach().numpy() for res in results]
        return label_list, scores_list

    def _detect_first_occurence(
        self, frame_idx_list: List[int], frame_list: Union[List[np.ndarray], List[str]]
    ) -> int:
        """Detect objects in frame_list until we find a frame with the desired object.

        Args:
            frame_idx_list: The list of indices to consider.
            frame_list: The List of images.

        Returns:
            The first index of frame_idx_list gets returned where the object was detected.
            If no object was detected -1 gets returned.
        """
        print(frame_idx_list)
        batch_size = self.params.get("batch_size", 10)
        for batch in tqdm(batch_list(frame_idx_list, batch_size)):
            batch_img_list = [process_input(frame_list[idx]) for idx in batch]
            label_list, scores_list = self._object_detection(batch_img_list)
            for frame_idx, label, score in zip(batch, label_list, scores_list):
                print(frame_idx, label, score)
                if len(label) != 0:
                    return frame_idx
        return -1

    def detect_sequences(self, frame_list: List[np.ndarray], offload_model: bool = True) -> List[BaseSequence]:
        """Detects frame sequences which contain painting content.

        Returns one BaseSequence indicating the painting content.
        A BaseSequence is defined by a start index and end frame index.

        Args:
            frame_list: List of frames in cv2 image format.
            offload_model: If set to true offloads the model to cpu afterwards.
                If several inferences are done after each other this can be set to False.

        Returns:
            List of BaseSequence objects with one entry.
        """
        text = self.params.get("prompt", "a hand")
        device = self.params.get("device", "cuda")
        max_frames_to_consider = self.params.get("max_frames_to_consider", 200)
        frame_steps = self.params.get("frame_steps", 10)

        if self.model is None or self.processor is None:
            raise ValueError(
                "Model and Processor not correctly instanciated. Make sure to call set_parameters to laod the model and processor."
            )

        self.model.to(device)

        frame_idx_list = list(range(len(frame_list)))
        # Start Frames to consider:
        frames_to_consider = frame_idx_list[::frame_steps][:max_frames_to_consider]
        start_index = self._detect_first_occurence(frame_idx_list=frames_to_consider, frame_list=frame_list)
        if start_index == -1:
            start_index = 0

        # End Frames to consider:
        frames_to_consider = frame_idx_list[::-frame_steps][:max_frames_to_consider]
        end_index = self._detect_first_occurence(frame_idx_list=frames_to_consider, frame_list=frame_list)
        if end_index == -1:
            end_index = len(frame_list) - 1

        if offload_model:
            self.model.to("cpu")

        return [BaseSequence(start_idx=start_index, end_idx=end_index, desc=f'Detected "{text}" in the video.')]

    def detect_sequences_on_disk(
        self, frame_path: Union[str, List[str]], offload_model: bool = True
    ) -> List[BaseSequence]:
        """
        Detects frame sequences which contain painting content.

        Returns one BaseSequence indicating the painting content.
        A BaseSequence is defined by a start index and end frame index.

        Args:
            frame_path:
                If it is a string than it gets interpreted as a cv2.VideoCapture filepath, this
                can be either an open video file or image file sequence.
                If it is a List of strings it gets interpredted as an image file sequence.
            offload_model: If set to true offloads the model to cpu afterwards.
                If several inferences are done after each other this can be set to False.

        Returns:
            List of BaseSequence objects with one entry.
        """
        text = self.params.get("prompt", "a hand")
        device = self.params.get("device", "cuda")
        max_frames_to_consider = self.params.get("max_frames_to_consider", 200)
        frame_steps = self.params.get("frame_steps", 10)

        if self.model is None or self.processor is None:
            raise ValueError(
                "Model and Processor not correctly instanciated. Make sure to call set_parameters to laod the model and processor."
            )

        self.model.to(device)

        if isinstance(frame_path, List):
            frame_list = frame_path
        else:
            with video_capture_context(video_path=frame_path) as cap:
                if not cap.isOpened():
                    raise ValueError(f"Was not able to read from {frame_path}.")
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # A placeholder, will populate afterwards
                frame_list = [np.zeros(0)] * num_frames

        frame_idx_list = list(range(len(frame_list)))
        frames_to_consider = frame_idx_list[::frame_steps][:max_frames_to_consider]
        frames_to_consider_bckwrds = frame_idx_list[::-frame_steps][:max_frames_to_consider]

        # If we have a video/frame list
        if isinstance(frame_path, str):
            with video_capture_context(video_path=frame_path) as cap:
                if not cap.isOpened():
                    raise ValueError(f"Was not able to read from {frame_path}.")
                for frame_idx in frames_to_consider + frames_to_consider_bckwrds:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    res, frame = cap.read()
                    if not res:
                        raise ValueError(f"Was not able to read frame with index {frame_idx} from {frame_path}.")
                    # Populate the frame list with all needed frames.
                    frame_list[frame_idx] = frame

        # Start Frames to consider:
        start_index = self._detect_first_occurence(frame_idx_list=frames_to_consider, frame_list=frame_list)
        if start_index == -1:
            start_index = 0

        # End Frames to consider:
        end_index = self._detect_first_occurence(frame_idx_list=frames_to_consider_bckwrds, frame_list=frame_list)
        if end_index == -1:
            end_index = len(frame_list) - 1

        if offload_model:
            self.model.to("cpu")

        return [BaseSequence(start_idx=start_index, end_idx=end_index, desc=f'Detected "{text}" in the video.')]

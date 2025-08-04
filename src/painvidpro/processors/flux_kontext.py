"""Class for the Loomis Keyframe detection."""

import logging
import os
from os.path import isfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from diffusers import FluxKontextPipeline
from PIL import Image
from tqdm import tqdm

from painvidpro.object_detection.factory import ObjectDetectionBase, ObjectDetectionFactory
from painvidpro.processors.keyframe import ProcessorKeyframe
from painvidpro.utils.image_processing import pil_resize_with_padding, pil_reverse_resize_with_padding
from painvidpro.utils.list_processing import batch_list
from painvidpro.utils.metadata import load_metadata, save_metadata
from painvidpro.video_processing.utils import video_capture_context, video_writer_context


class ProcessorFluxKontext(ProcessorKeyframe):
    def __init__(self):
        """Class to process videos."""
        super().__init__()
        self._canvas_detector: Optional[ObjectDetectionBase] = None
        self._flux_kontext_pipe: Optional[FluxKontextPipeline] = None

    @property
    def canvas_detector(self) -> ObjectDetectionBase:
        if self._canvas_detector is None:
            raise RuntimeError(
                (
                    "Canvas Detector not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._canvas_detector

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
        detect_canvas = self.params.get("detect_canvas", False)
        try:
            if detect_canvas:
                self._canvas_detector = ObjectDetectionFactory().build(
                    self.params["canvas_detector_algorithm"], self.params["canvas_detector_config"]
                )
            self._flux_kontext_pipe = FluxKontextPipeline.from_pretrained(
                self.params["flux_kontext_algorithm"], torch_dtype=torch.bfloat16
            )
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        super().set_default_parameters()
        self.params["yt_video_format"] = "bestvideo[height<=480]"
        self.params["detect_canvas"] = False
        self.params["canvas_detector_algorithm"] = "ObjectDetectionGroundingDino"
        self.params["canvas_detector_config"] = {"prompt": "a canvas."}
        self.params["flux_kontext_algorithm"] = "black-forest-labs/FLUX.1-Kontext-dev"
        self.params["flux_kontext_config"] = {
            "prompt": "Remove hand. Remove pencil.",
            "prompt_watermark": "Remove watermark",
            "prompt_lighting": "Flat lighting of the painting.",
            "batch_size": 1,
        }
        self.params["remove_logos"] = False
        self.params["detect_keyframes"] = False

    def detect_canvas(self, video_dir: Path, video_path: str, metadata: Dict[str, Any]) -> bool:
        """Detects a canvas region in the video, crops the video to this region, and updates metadata.

        If the metadata indicates the canvas has already been detected, the function exits early. Otherwise,
        it processes the video starting from the frame specified in the metadata to detect the canvas.
        If a canvas is detected, the video is cropped to the detected region,
        overwriting the original video file. Metadata is updated to reflect the detection status.

        Args:
            video_dir: Path to the directory containing the video.
            video_path: Path to the video file.
            metadata: Metadata dictionary containing:
                      - "start_frame_idx": Frame index to start canvas detection.
                      - "canvas_detected": Flag indicating if the canvas was already detected.
                      The dictionary is modified in-place to add/update the "canvas_detected" key.

        Returns:
            bool: Always returns True, indicating the process completed (successful detection or fallback to original video).
                  Returns immediately if the canvas was already pre-detected.

        Side Effects:
            - Overwrites the original video with the cropped version if a canvas is detected.
            - Modifies the metadata dictionary to set "canvas_detected" = True on successful detection.
            - Persists updated metadata to disk.
        """
        # In case we already detected the canvas
        if metadata.get("canvas_detected", False):
            return True
        start_frame_idx = metadata["start_frame_idx"]

        with video_capture_context(video_path=video_path) as cap:
            frame_idx_to_consider = iter([start_frame_idx])
            selected_frames: List[np.ndarray] = []

            frame_idx = 0
            frame_idx_to_select = next(frame_idx_to_consider, None)
            ret, frame = cap.read()
            while ret and frame_idx_to_select is not None:
                if frame_idx_to_select == frame_idx:
                    selected_frames.append(frame)
                    frame_idx_to_select = next(frame_idx_to_consider, None)
                ret, frame = cap.read()
                frame_idx += 1

            # Detect the objects
            object_list: List[Any] = []
            for entry in self.canvas_detector.detect_objects(selected_frames):
                for det_obj in entry:
                    object_list.append(det_obj)
            object_list.sort(key=lambda x: x["score"], reverse=True)

        if len(object_list) == 0:
            self.logger.info("Was not able to detect a canvas in the video, using the original video for processing.")
            return True

        cavas_only_video_path = str(video_dir / "canvas_only_video.mp4")
        with video_capture_context(video_path=video_path) as cap:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Take the box with highest score
            x_min, y_min, x_max, y_max = object_list[0]["box"]

            # Convert to integers (pixel indices)
            x_min = max(0, int(round(x_min)))
            y_min = max(0, int(round(y_min)))
            x_max = min(width, int(round(x_max)))
            y_max = min(height, int(round(y_max)))
            new_width = x_max - x_min
            new_height = y_max - y_min

            with video_writer_context(
                output_path=cavas_only_video_path, width=new_width, height=new_height, fps=fps
            ) as vid_out:
                ret, frame = cap.read()
                while ret:
                    # Crop the image
                    cropped_image = frame[y_min:y_max, x_min:x_max]
                    vid_out.write(cropped_image)
                    ret, frame = cap.read()

        # Overwrite the original video with the cropped one
        os.replace(src=cavas_only_video_path, dst=video_path)
        metadata["canvas_detected"] = True
        save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)
        self.logger.info(
            (
                f"Detected Canvas with coordinates ({x_min}, {y_min}), ({x_max}, {y_max}).\n"
                "The original video was overwritten with the cropped version."
            )
        )
        return True

    def _estimate_num_frames(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        fps: int,
    ) -> int:
        """Compute the number of frames to sample.

        Selects a frame every 3 seconds at most 1000 frames.

        Args:
            start_frame_idx: The starting frame index to consider.
            end_frame_idx: The ending frame index to consider.
            fps: Frames per second of the video.

        Returns:
            Number of frames to sample.
        """
        num_frames = (end_frame_idx - start_frame_idx) // (3 * fps)
        num_frames = min(num_frames, 1000)

        self.logger.info(f"Selected num_frames: {num_frames}.")

        return num_frames

    def extract_frames(
        self,
        video_dir: Path,
        video_path: str,
        metadata: Dict[str, Any],
        num_frames: int = -1,
        disable_tqdm: bool = True,
    ) -> bool:
        """Extracts frames from the video and stores them on disk.

        Extracts num_frames by choosing samples from the video and removing occlusions.

        Does nothing if metadata contains `extracted_frames` field.

        Args:
            video_dir: Path to the directory containing the video.
            video_path: Path to the video file.
            metadata: Metadata dict containing 'start_frame_idx' and 'end_frame_idx'.
            num_frames: The number of frames to extract from the video.
            disable_tqdm: If True, disables the tqdm progress bar.

        Returns:
            A bool indicating success. If False it can be:
            - If `end_frame_idx` is smaller than `start_frame_idx`.
            - If reading a frame from the video fails.
        """
        if "extracted_frames" in metadata:
            return True

        start_frame_idx = metadata["start_frame_idx"]
        end_frame_idx = metadata["end_frame_idx"]
        extr_frame_dir = video_dir / self.extr_folder_name
        extr_frame_dir.mkdir(parents=True, exist_ok=True)
        if start_frame_idx < 0:
            self.logger.info(f"The start_frame_idx {start_frame_idx} must be at least 0.")
            return False
        if end_frame_idx < start_frame_idx:
            self.logger.info(
                f"The end_frame_idx {end_frame_idx} must be bigger than start_frame_idx {start_frame_idx}."
            )
            return False

        try:
            self._flux_kontext_pipe = self.flux_kontext_pipe.to("cuda")
            batch_size = self.params["flux_kontext_config"].get("batch_size", 1)
            kontext_prompt = self.params["flux_kontext_config"].get("prompt", "Remove hand. Remove pencil.")
            with video_capture_context(video_path=video_path) as cap:
                # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if end_frame_idx > num_frames_video:
                    self.logger.info(
                        f"The end_frame_idx {end_frame_idx} must be smaller equals than the total number of frames in the video {num_frames_video}. Setting it now to {num_frames_video}."
                    )
                    end_frame_idx = num_frames_video
                if num_frames < 0:
                    num_frames = self._estimate_num_frames(
                        start_frame_idx=start_frame_idx, end_frame_idx=end_frame_idx, fps=int(fps)
                    )

                num_frames = min(num_frames, end_frame_idx - start_frame_idx + 1)
                selected_frame_indices = np.linspace(start_frame_idx, end_frame_idx, num_frames, dtype=int).tolist()
                frame_paths: List[str] = []
                ret_frame_idx: List[int] = []
                frame_idx = 0
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Was not able to read first frame from {video_path}.")

                # Read the sampled frames
                for batched_frame_indices in tqdm(
                    batch_list(selected_frame_indices, batch_size=batch_size),
                    desc="Removing occlusions",
                    disable=disable_tqdm,
                ):
                    batch_frame_list: List[Image.Image] = []
                    for selected_f_idx in batched_frame_indices:
                        while frame_idx < selected_f_idx:
                            ret, frame = cap.read()
                            if not ret:
                                # Something went wrong
                                raise ValueError(f"Reading frame with index {frame_idx} failed.")
                            frame_idx += 1

                        if frame is None:
                            raise ValueError(f"Was not able to get frame with index {selected_f_idx}.")
                        batch_frame_list.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

                    # Resize and pad to 1024x1024 pixels for better peroformance with Flux Kontext
                    batch_finput_list = [
                        pil_resize_with_padding(frame, target_size=(1024, 1024)) for frame in batch_frame_list
                    ]
                    kontext_image_list = self.flux_kontext_pipe(
                        image=batch_finput_list, prompt=[kontext_prompt] * len(batch_finput_list), guidance_scale=2.5
                    ).images
                    # Crop and Resize to original image
                    kontext_image_list = [
                        pil_reverse_resize_with_padding(frame, og_frame.size)
                        for frame, og_frame in zip(kontext_image_list, batch_frame_list)
                    ]
                    for frame_idx, kontext_image in zip(batched_frame_indices, kontext_image_list):
                        frame_name = f"frame_{str(frame_idx).zfill(self.zfill_num)}.png"
                        frame_path = str(extr_frame_dir / frame_name)
                        kontext_image.save(frame_path)
                        ret_frame_idx.append(frame_idx)
                        frame_paths.append(frame_path)

            # Save extracted frames to metadata
            metadata_addition: List[Dict[str, Any]] = []
            for median_frame_idx, median_frame_path in zip(ret_frame_idx, frame_paths):
                metadata_addition.append(
                    {
                        "index": median_frame_idx,
                        "path": median_frame_path,
                        "extraction_method": "flux_kontext",
                    }
                )
            metadata["extracted_frames"] = metadata.get("extracted_frames", []) + metadata_addition
            save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)
            self._flux_kontext_pipe = self.flux_kontext_pipe.to("cpu")
        except Exception as e:
            self.logger.info(f"Was not able to extract frames from {video_path}: {e}")
            return False

        self.logger.info(f"Successfully extracted {len(frame_paths)} frames with Flux-Kontext.")
        return True

    def _create_combined_mask(self, mask_list: List[np.ndarray]) -> np.ndarray:
        """Extracts a mask where pixels are masked at least half the time."""
        # Convert the list of boolean masks to a 3D numpy array
        mask_array = np.array(mask_list)
        sum_masks = mask_array.sum(axis=0)
        # Calculate the threshold (n/2 rounded up)
        n = len(mask_list)
        threshold = (n + 1) // 2
        # Create the combined mask where count >= threshold
        combined_mask = sum_masks >= threshold
        return combined_mask

    def _remove_logo(self, video_dir: Path, metadata: Dict[str, Any], disable_tqdm: bool = True) -> bool:
        """Removes logos from extracted frames.

        Processes all frames listed in metadata's "extracted_frames" by using Flux-Kontext.

        Args:
            video_dir: Directory containing extracted frames from video.
            metadata: Metadata as dict.
            disable_tqdm: If True, disables progress bar visualization.

        Returns:
            bool: Always returns True, indicating completion of processing attempts for all frames.
                  Individual frame failures are logged but don't prevent overall completion.
        """
        extracted_frame_list: List[str] = metadata.get("extracted_frames", [])
        frame_path_list: List[str] = []

        try:
            self._flux_kontext_pipe = self.flux_kontext_pipe.to("cuda")
            batch_size = self.params["flux_kontext_config"].get("batch_size", 1)
            kontext_prompt = self.params["flux_kontext_config"].get("prompt_watermark", "Remove watermark")
            for batched_frame_entries in tqdm(
                batch_list(extracted_frame_list, batch_size=batch_size),
                desc="Removing logos and watermakrs",
                disable=disable_tqdm,
            ):
                frame_path_list = [str(video_dir / entry["path"]) for entry in batched_frame_entries]
                frame_list = [Image.open(frame_path) for frame_path in frame_path_list]

                # Resize and pad to 1024x1024 pixels for better peroformance with Flux Kontext
                batch_finput_list = [pil_resize_with_padding(frame, target_size=(1024, 1024)) for frame in frame_list]
                kontext_image_list = self.flux_kontext_pipe(
                    image=batch_finput_list, prompt=[kontext_prompt] * len(batch_finput_list), guidance_scale=2.5
                ).images
                # Crop and Resize to original image
                kontext_image_list = [
                    pil_reverse_resize_with_padding(frame, og_frame.size)
                    for frame, og_frame in zip(kontext_image_list, frame_list)
                ]

                for frame_path, kontext_image in zip(frame_path_list, kontext_image_list):
                    kontext_image.save(frame_path)

            self._flux_kontext_pipe = self.flux_kontext_pipe.to("cpu")
        except Exception as e:
            self.logger.info((f"Was not able to remove Logos." f"Following error occurred:\n{e}"))

        self.logger.info(("Successfully removed logos in frames."))
        return True

    def save_reference_frame(self, video_dir: Path, metadata: Dict[str, Any], reference_frame_path: str) -> bool:
        """Saves the last extracted frame as reference frame.

        Args:
            video_dir: Path to the directory containing the video.
            metadata: The metadata.
            reference_frame_path: The path to the reference frame.

        Returns:
            A bool indicating if the saving was successfull.
        """
        if isfile(reference_frame_path):
            return True

        if "extracted_frames" not in metadata:
            self.logger.info("Tried to save reference frame but no extracted_frames were found.")
            return False

        max_index = -1
        selected_frame = None
        for extracted_frame in metadata["extracted_frames"]:
            if extracted_frame["index"] > max_index:
                max_index = extracted_frame["index"]
                selected_frame = extracted_frame

        if selected_frame is None:
            self.logger.info("Tried to save reference frame but no extracted_frames were found.")
            return False

        try:
            selected_frame_path = str(video_dir / selected_frame["path"])
            cv2.imwrite(reference_frame_path, cv2.imread(selected_frame_path))
        except Exception as e:
            self.logger.info(
                (
                    f"Tried to save reference frame to {reference_frame_path}"
                    f" but was not able to load image from video dir {video_dir}"
                    f" with relative path {selected_frame.get('path', "no path was found")}."
                    f"\n{e}"
                )
            )
            return False

        self.logger.info(f"Successfully saved reference frame to {reference_frame_path}.")
        return True

    def process(self, video_dir_list: List[str], batch_size: int = -1) -> List[bool]:
        """Extracts the Keyframes of a Loomis Portrait video.

        The processor downloads the video, splits it into two (real image and drawing),
        detects the start/end frame in the video and extracts keyframes between
        start/end frame.

        Args:
            video_dir_list: List of paths where the videos are stored.
            batch_size: The batch size for the keyframe detection.
                batch_size < 0, do not set the batch size explicitly.
                batch_size > 0, set the patch size.

        Returns:
            A list of bools, indidcating for each element in video_dir_list, if the processing
            was successfull.
        """
        ret = [False] * len(video_dir_list)
        disable_tqdm = self.params.get("disable_tqdm", True)
        detect_canvas = self.params.get("detect_canvas", True)
        remove_logos = self.params.get("remove_logos", False)
        for i, vd in enumerate(video_dir_list):
            video_dir = Path(vd)
            log_file = str(video_dir / "ProcessorFluxKontext.log")
            logging.basicConfig(
                filename=log_file,
                filemode="w",
                force=True,
            )
            video_file_path = str(video_dir / self.video_file_name)
            reference_frame_path = str(video_dir / self.reference_frame_name)

            # Loading the metadata dict
            succ, metadata = load_metadata(video_dir, metadata_name=self.metadata_name)
            if not succ:
                self.logger.info(f" Failed opening metadata {str(video_dir / self.metadata_name)}.")
                continue

            # Downloading the video
            if not self._download_video(video_file_path=video_file_path, metadata=metadata):
                continue

            # Detecting start and end frame
            if not self._detect_start_end_frame(
                video_dir=video_dir, video_file_path=video_file_path, metadata=metadata, batch_size=batch_size
            ):
                continue

            # Detect the canvas if specified
            if detect_canvas and not self.detect_canvas(
                video_dir=video_dir, video_path=video_file_path, metadata=metadata
            ):
                continue

            # Extracting frames by computing the median of sampled frames
            if not self.extract_frames(
                video_dir=video_dir,
                video_path=video_file_path,
                metadata=metadata,
                disable_tqdm=disable_tqdm,
            ):
                continue

            # Removes logo and other text from extracted frames
            if remove_logos and not self._remove_logo(
                video_dir=video_dir, metadata=metadata, disable_tqdm=disable_tqdm
            ):
                continue

            # Save the reference frame
            if not self.save_reference_frame(
                video_dir=video_dir, metadata=metadata, reference_frame_path=reference_frame_path
            ):
                continue

            # Post processing and cleaning up
            if not self._post_process(video_file_path=video_file_path):
                continue

            # Processing was successfull
            metadata["processed_video_name"] = self.video_file_name
            metadata["processed"] = True
            save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)
            ret[i] = True

        # Clear file logging
        logging.basicConfig(
            filename=None,
            force=True,
        )

        return ret

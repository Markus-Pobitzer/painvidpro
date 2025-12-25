"""Class for the Loomis Keyframe detection."""

import logging
import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

from painvidpro.data_storage.hdf5_video_archive import DynamicVideoArchive
from painvidpro.logging.logging import setup_logger
from painvidpro.occlusion_removing.factory import OcclusionRemovingBase, OcclusionRemovingFactory
from painvidpro.processors.base import ProcessorBase
from painvidpro.utils.image_processing import process_input
from painvidpro.video_processing.utils import video_capture_context, video_writer_context
from painvidpro.video_processing.youtube import download_video


class ProcessorSAM3(ProcessorBase):
    def __init__(self):
        """Class to process videos."""
        super().__init__()
        self.set_default_parameters()
        self.logger = setup_logger(name=__name__)
        self._sam3_model: Optional[Sam3Model] = None
        self._sam3_processor: Optional[Sam3Processor] = None
        self.video_file_name = "video.mp4"
        self.frame_data = "frame_data.h5"
        self._logo_removing_model: Optional[OcclusionRemovingBase] = None

    @property
    def sam3_model(self) -> Sam3Model:
        if self._sam3_model is None:
            raise RuntimeError(
                (
                    "Model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._sam3_model

    @property
    def sam3_processor(self) -> Sam3Processor:
        if self._sam3_processor is None:
            raise RuntimeError(
                (
                    "Processor not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._sam3_processor

    @property
    def logo_removing_model(self) -> OcclusionRemovingBase:
        if self._logo_removing_model is None:
            raise RuntimeError(
                (
                    "Logo REmoving model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model."
                )
            )
        return self._logo_removing_model

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        self.params.update(params)

        sam3_model = self.params.get("sam3_model", "facebook/sam3")
        self._detect_canvas = self.params.get("detect_canvas", False)
        self.remove_logos = self.params.get("remove_logos", False)
        self.device = self.params.get("device", "cuda")
        try:
            self._sam3_model = Sam3Model.from_pretrained(sam3_model).to(self.device)
            self._sam3_processor = Sam3Processor.from_pretrained(sam3_model)
            self._logo_removing_model = OcclusionRemovingFactory().build(
                self.params["logo_removing_algorithm"], self.params["logo_removing_config"]
            )
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "yt_video_format": "bestvideo[height<=480]",
            "sam3_model": "facebook/sam3",
            "sam3_config": {},
            "occlusion_masking_config": {"prompt": ["a hand", "a paintbrush", "a pencil"]},
            "disable_tqdm": True,
            "remove_videos_after_processing": False,
            "start_end_frame_detector_config": {"prompt": "a hand"},
            "device": "cuda",
            "detect_canvas": True,
            "canvas_detector_config": {"prompt": "a blank canvas"},
            "remove_logos": False,
            "logo_masking_config": {"prompt": "a logo"},
            "logo_removing_algorithm": "OcclusionRemovingLamaInpainting",
            "logo_removing_config": {},
            "overwrite_with_median_frame": True,
        }

    def _download_video(self, video_file_path: str, frame_data: DynamicVideoArchive) -> bool:
        """Downloads the video if not alredy downloaded.

        Args:
            video_file_path: Path to the video file on disk.
            frame_data: Frame data to update metadata.

        Returns:
            bool: True if successfull or already exists, False otherwise.
        """
        try:
            if not os.path.isfile(video_file_path):
                with frame_data:
                    # TODO: Check which site it is from needs to be done dynamically
                    if "youtube" in video_file_path:
                        url: str = frame_data.get_global_metadata()["id"]  # type: ignore
                        video_format = self.params.get("yt_video_format", "bestvideo[height<=480]")
                        try:
                            yt_dlp_retcode = download_video(url, video_file_path, format=video_format)
                            if yt_dlp_retcode != 0:
                                self.logger.info(
                                    (
                                        " While downloading the video an error occured.\n"
                                        " The used library for downloading is https://github.com/yt-dlp/yt-dlp\n"
                                        f" The return code of yt-dlp was {yt_dlp_retcode}."
                                    )
                                )
                                return False
                        except Exception as e:
                            self.logger.info(
                                (
                                    f" Failed downloading YouTube video with video Id {url}"
                                    f" and video format {video_format} to {video_file_path}:\n"
                                    f"{e}\n"
                                )
                            )
                            return False
                    else:
                        self.logger.info(
                            (
                                f" Was not able to determine how to download video to {video_file_path}.\n"
                                f"The metadata: {frame_data.get_global_metadata()}."
                            )
                        )
                        return False

                    # Reset that canvas was detected when the video gets downloaded.
                    frame_data.set_global_metadata("canvas_detected", False)
        except Exception as e:
            self.logger.info(f" Failed downloading video to {video_file_path}: {e}")
            return False
        return True

    def _delete_video(self, video_file_path: str):
        """Deletes the video specifid at video_file_path.

        If no file was found at the specified path, nothing is done.
        Args:
            video_file_path: The path to the video to delte.
        """
        if os.path.isfile(video_file_path):
            os.remove(video_file_path)

    def _detect_start_end_frame(
        self,
        video_file_path: str,
        frame_data: DynamicVideoArchive,
        batch_size: int = -1,
        max_frames_to_consider: int = 300,
        frame_steps: int = 30,
    ) -> bool:
        """Detect start and end frame in the video if not already detected.

        Args:
            video_file_path: Path to the video on disk.
            frame_data: The DynamicVideoArchive.
            batch_size: If > 0, then set the batch size of sequence detector
                for detecting start and end frame.
            max_frames_to_conside: max number of frames to consider from the first
                and last instance.
            frame_steps: Take each frame_steps frame.

        Returns:
            Boolean indicating success.
        """
        with frame_data:
            metadata: Dict[str, Any] = frame_data.get_global_metadata()  # type: ignore

        if "start_frame_idx" not in metadata or "end_frame_idx" not in metadata:
            try:
                prompt = self.params["start_end_frame_detector_config"]["prompt"]
                start_idx = -1
                end_idx = -1
                with video_capture_context(video_path=video_file_path) as cap:
                    if not cap.isOpened():
                        raise ValueError(f"Was not able to read from {video_file_path}.")

                    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    frame_idx_list = list(range(num_frames))
                    frames_to_consider = frame_idx_list[::frame_steps][:max_frames_to_consider]
                    frames_to_consider_bckwrds = frame_idx_list[::-frame_steps][:max_frames_to_consider]

                    def _first_appearance(frame_idx_list: List[int]) -> int:
                        for frame_idx in frame_idx_list:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            res, frame = cap.read()
                            if not res:
                                self.logger.info(
                                    f"Was not able to read frame with index {frame_idx} from {video_file_path}."
                                )
                            frame = process_input(frame, convert_bgr_to_rgb=True)
                            inputs = self.sam3_processor(images=frame, text=prompt, return_tensors="pt").to(
                                self.device
                            )
                            with torch.no_grad():
                                outputs = self.sam3_model(**inputs)
                            results = self.sam3_processor.post_process_instance_segmentation(
                                outputs,
                                threshold=0.5,
                                mask_threshold=0.5,
                                target_sizes=inputs.get("original_sizes").tolist(),
                            )[0]["masks"]
                            if len(results) > 0:
                                return frame_idx
                        return -1

                    start_idx = _first_appearance(frame_idx_list=frames_to_consider)
                    end_idx = _first_appearance(frame_idx_list=frames_to_consider_bckwrds)

                with frame_data:
                    frame_data.set_global_metadata("start_frame_idx", start_idx)
                    frame_data.set_global_metadata("end_frame_idx", end_idx)
            except Exception as e:
                self.logger.info(f" Failed detecting start and end frame from {video_file_path}: {e}")
                return False

        return True

    def _post_process(self, video_file_path: str) -> bool:
        """Does post processing.

        Includes deletion of files if specified.

        Args:
            video_file_path: Path to the video on disk.

        Returns:
            A bool indicating success.
        """
        if self.params.get("remove_videos_after_processing"):
            self._delete_video(video_file_path=video_file_path)
        return True

    def detect_canvas(
        self, video_dir: Path, video_path: str, frame_data: DynamicVideoArchive, canvas_erosion: int = 2
    ) -> bool:
        """Detects a canvas region in the video, crops the video to this region, and updates metadata.

        If the metadata indicates the canvas has already been detected, the function exits early. Otherwise,
        it processes the video starting from the frame specified in the metadata to detect the canvas.
        If a canvas is detected, the video is cropped to the detected region,
        overwriting the original video file. Metadata is updated to reflect the detection status.

        Args:
            video_dir: Path to the directory containing the video.
            video_path: Path to the video file.
            frame_data: DynamicVideoArchive containting the metadata dictionary:
                      - "start_frame_idx": Frame index to start canvas detection.
                      - "canvas_detected": Flag indicating if the canvas was already detected.
                      The dictionary is modified add/update the "canvas_detected" key.
            canvas_erosion: Indicates by how many pixels the canvas should be eroded.

        Returns:
            bool: Always returns True, indicating the process completed (successful detection or fallback to original video).
                  Returns immediately if the canvas was already pre-detected.

        Side Effects:
            - Overwrites the original video with the cropped version if a canvas is detected.
            - Modifies the metadata dictionary to set "canvas_detected" = True on successful detection.
            - Persists updated metadata to disk.
        """
        with frame_data:
            metadata: Dict[str, Any] = frame_data.get_global_metadata()  # type: ignore
            # In case we already detected the canvas
            if metadata.get("canvas_detected", False):
                return True
            start_frame_idx = int(metadata["start_frame_idx"])

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

            # Detect the canvas
            frame = process_input(frame, convert_bgr_to_rgb=True)
            prompt = self.params.get("canvas_detector_config", {}).get("prompt", "a blank canvas")
            inputs = self.sam3_processor(images=frame, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.sam3_model(**inputs)
            bbox = (
                self.sam3_processor.post_process_instance_segmentation(
                    outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
                )[0]["boxes"]
                .cpu()
                .to(float)
                .numpy()
            )

            if len(bbox) > 0:
                # We take the first appearance
                x_min, y_min, x_max, y_max = bbox[0]
            else:
                self.logger.info(
                    "Was not able to detect a canvas in the video, using the original video for processing."
                )
                return True

        cavas_only_video_path = str(video_dir / "canvas_only_video.mp4")
        with video_capture_context(video_path=video_path) as cap:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Convert to integers (pixel indices)
            x_min = max(0, int(round(x_min) + canvas_erosion))
            y_min = max(0, int(round(y_min) + canvas_erosion))
            x_max = min(width, int(round(x_max) - canvas_erosion))
            y_max = min(height, int(round(y_max) - canvas_erosion))
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
        with frame_data:
            frame_data.set_global_metadata("canvas_detected", True)
            frame_data.set_global_metadata("canvas_coordinates", [x_min, y_min, x_max, y_max])
        self.logger.info(
            (
                f"Detected Canvas with coordinates ({x_min}, {y_min}), ({x_max}, {y_max}).\n"
                "The original video was overwritten with the cropped version."
            )
        )
        return True

    def _extract_frame(
        self,
        frame: np.ndarray,
        prev_extr_frame: Optional[np.ndarray] = None,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """Extracts the frame on the given inputs.

        Args:
            frame: Sampled frame with shape [H, W, 3].
            prev_extr_frame: The previously extracted frame with shape [H, W, 3].
                If prev_extr_frame is not set, than we inpaint the region.
            kernel_size: Kernel size for occlusion mask dilation.

        Returns:
            The median frame with shape [H, W, 3], computed while ignoring masked regions.
            In RGB format.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Detect the occlusions
        frame = process_input(frame, convert_bgr_to_rgb=True)
        prompt = self.params.get("occlusion_masking_config", {}).get("prompt", ["a hand", "a paintbrush", "a pencil"])
        if isinstance(prompt, str):
            prompt = [prompt]
        frame_list = [frame] * len(prompt)
        # Could run OOM if to many prompt items
        inputs = self.sam3_processor(images=frame_list, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam3_model(**inputs)
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
        )

        # Extract masks from result. Each index in the list corresponds to a prompt.
        empty_mask = np.zeros(frame.shape[:2], dtype=np.bool)
        mask_list: List[np.ndarray] = [empty_mask] + [
            res["masks"].cpu().numpy().any(axis=0) for res in results if len(res["masks"]) != 0
        ]
        # Combine masks of all prompts
        mask = np.stack(mask_list, axis=0).any(axis=0)
        # Dilate for better coverage
        mask = cv2.dilate(np.uint8(mask), kernel, iterations=1).astype(bool)

        if prev_extr_frame is not None:
            mask = mask[..., np.newaxis]
            ret_frame = np.where(mask, prev_extr_frame, frame)
        else:
            # Notation a bit misleading but here we convert from rgb to bgr (it is just a channel switch).
            bgr_frame = process_input(frame, convert_bgr_to_rgb=True)
            ret_frame = self.logo_removing_model.remove_occlusions(
                frame_list=[bgr_frame], mask_list=[mask], offload_model=False
            )[0]
        return ret_frame

    def extract_n_frames(
        self,
        video_path: str,
        frame_data: DynamicVideoArchive,
        num_frames: int = 1000,
        kernel_size: int = 5,
        disable_tqdm: bool = True,
    ) -> bool:
        """Extracts N frames from the video and stores them on disk.

        Does nothing if metadata contains `extracted_frames` field.

        Args:
            video_path: Path to the video file.
            metadata: Metadata dict containing 'start_frame_idx' and 'end_frame_idx'.
            num_frames: The number of frames to extract.
            kernel_size: Size of the dilation kernel applied to occlusion masks.
            disable_tqdm: If True, disables the tqdm progress bar.

        Returns:
            A bool indicating success. If False it can be:
            - If `end_frame_idx` is smaller than `start_frame_idx`.
            - If reading a frame from the video fails.
            - If no frame is read for a given frame sequence.
        """
        with frame_data:
            if len(frame_data) > 0:
                self.logger.info(f"Already found {len(frame_data)} frames in the dataset, skipping frame extraction.")
                return True

            metadata: Dict[str, Any] = frame_data.get_global_metadata()  # type: ignore
            start_frame_idx = int(metadata["start_frame_idx"])
            end_frame_idx = int(metadata["end_frame_idx"])

        if end_frame_idx < start_frame_idx:
            self.logger.info(
                f"The end_frame_idx {end_frame_idx} must be bigger than start_frame_idx {start_frame_idx}."
            )
            return False

        processed_frame_ids = []
        try:
            with video_capture_context(video_path=video_path) as cap:
                num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                num_frames = min(num_frames, num_frames_video)

                frame_indices = np.linspace(start=start_frame_idx, stop=end_frame_idx, num=num_frames).tolist()
                frame_idx = 0
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Was not able to read first frame from {video_path}.")

                prev_frame: Optional[np.ndarray] = None
                for selected_frame_idx in tqdm(frame_indices, desc="Extracting frames", disable=disable_tqdm):
                    while frame_idx < selected_frame_idx:
                        ret, frame = cap.read()
                        if not ret:
                            # Something went wrong
                            raise ValueError(f"Reading frame with index {frame_idx} failed.")
                        frame_idx += 1

                    if frame is None:
                        raise ValueError(f"Was not able to get frame with index {selected_frame_idx}.")

                    cleaned_frame = self._extract_frame(
                        frame=frame, prev_extr_frame=prev_frame, kernel_size=kernel_size
                    )

                    frame_progress = (frame_idx - start_frame_idx) / end_frame_idx
                    with frame_data:
                        frame_data.add_frame(
                            cleaned_frame,
                            {
                                "processor": __name__,
                                "frame_index": frame_idx,
                                "frame_progress": frame_progress,
                                "logo_removed": False,
                            },
                        )
                    prev_frame = cleaned_frame
                    processed_frame_ids.append(frame_idx)

        except Exception as e:
            self.logger.info(f"Was not able to extract frames from {video_path}: {e}")
            return False

        self.logger.info(f"Successfully extracted {len(processed_frame_ids)} frames.")
        return True

    def overwrite_with_median_frame(self, frame_data: DynamicVideoArchive, window_size=11) -> bool:
        """Applies a sliding window median computation over the frames.

        Args:
            frame_data: The DynamicVideoArchive.
            window_size: The size of the sliding window.

        Returns:
            A bool indicating success.
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd.")

        # radius = window_size // 2

        with frame_data:
            num_frames = len(frame_data)
            if num_frames <= window_size:
                # Edge case we do nothing
                return True

            index = 1
            # We assume a small windowsize
            frame_list = [np.asarray(img) for img in [frame_data[i] for i in range(window_size)]]
            while index * 2 + 1 < window_size:
                current_window = index * 2 + 1
                median_array = np.median(np.stack(frame_list[:current_window], axis=0), axis=0).astype(np.uint8)
                metadata = frame_data.get_frame_metadata(index)
                frame_data.update_frame(index=index, image=median_array, metadata_dict=metadata)
                index += 1

            # Compute sliding window for the normal cases
            window = deque(np.stack(frame_list, axis=0), maxlen=window_size)
            for idx in range(window_size, num_frames):
                median_array = np.median(np.stack(window), axis=0).astype(window[0].dtype)
                metadata = frame_data.get_frame_metadata(index)
                frame_data.update_frame(index=index, image=median_array, metadata_dict=metadata)

                window.append(np.array(np.asarray(frame_data[idx])))
                index += 1

            while index < num_frames - 1:
                median_array = np.median(np.stack(window), axis=0).astype(window[0].dtype)
                metadata = frame_data.get_frame_metadata(index)
                frame_data.update_frame(index=index, image=median_array, metadata_dict=metadata)

                # Here we remove to entries on the left since no new entries can be added on the right
                window.popleft()
                window.popleft()
                index += 1
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

    def _remove_logo(self, frame_data: DynamicVideoArchive, disable_tqdm: bool = True) -> bool:
        """Removes logos from extracted frames.

        Processes all frames listed in metadata's "extracted_frames" by first detecting logos
        with a masking model, then removing them. Overwrites original frames
        with cleaned versions.

        Args:
            frame_data: DynamicVideoArchive containing frame data.
            disable_tqdm: If True, disables progress bar visualization.

        Returns:
            bool: Always returns True, indicating completion of processing attempts for all frames.
                  Individual frame failures are logged but don't prevent overall completion.
        """
        succ_count = 0
        mask_list: List[np.ndarray] = []

        frame_idx_list = []
        with frame_data:
            num_frames = len(frame_data)
            for frame_idx in range(num_frames):
                metadata_dict = frame_data.get_frame_metadata(frame_idx)
                if "logo_removed" not in metadata_dict or not metadata_dict["logo_removed"]:
                    frame_idx_list.append(frame_idx)

        if len(frame_idx_list) == 0:
            self.logger.info(
                "No logos were removed, either no frames found or all frames had already their logo removed."
            )
            return True

        # Detection
        for frame_idx in tqdm(frame_idx_list, disable=disable_tqdm, desc="Detecting Logos from frames"):
            try:
                with frame_data:
                    # numpy array in RGB ordering
                    frame = np.asarray(frame_data.get_frame(frame_idx)).copy()

                prompt = self.params.get("logo_masking_config", {}).get("prompt", ["a logo"])
                if isinstance(prompt, str):
                    prompt = [prompt]
                frame_list = [frame] * len(prompt)
                # Could run OOM if to many prompt items
                inputs = self.sam3_processor(images=frame_list, text=prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.sam3_model(**inputs)
                results = self.sam3_processor.post_process_instance_segmentation(
                    outputs, threshold=0.5, mask_threshold=0.5, target_sizes=inputs.get("original_sizes").tolist()
                )

                # Extract masks from result. Each index in the list corresponds to a prompt.
                empty_mask = np.zeros(frame.shape[:2], dtype=np.bool)
                frame_mask_list: List[np.ndarray] = [empty_mask] + [
                    res["masks"].cpu().numpy().any(axis=0) for res in results if len(res["masks"]) != 0
                ]
                # Extract a single mask for the entry
                mask_list.append(np.stack(frame_mask_list, axis=0).any(axis=0))  # type: ignore
            except Exception as e:
                self.logger.info(
                    (
                        f"Was not able to detect Logos of frame with index {frame_idx}. "
                        f"Following error occurred:\n{e}"
                    )
                )

        if len(mask_list) == 0:
            self.logger.info("No frames left to remove logos from.")
            return True

        # Computing the mask where fix logos most likely are
        video_mask = self._create_combined_mask(mask_list=mask_list)
        mask_list = [np.logical_or(mask, video_mask) for mask in mask_list]

        # Removing
        for frame_idx, mask in tqdm(
            zip(frame_idx_list, mask_list), disable=disable_tqdm, desc="Removing Logos from frames"
        ):
            try:
                with frame_data:
                    # PIL image in RGB ordering
                    pil_image = frame_data.get_frame(frame_idx)
                    # Bad notation, in this case it converts from RGB to BGR
                    frame = process_input(np.asarray(pil_image).copy(), convert_bgr_to_rgb=True)

                cleaned_frame = self.logo_removing_model.remove_occlusions(
                    frame_list=[frame], mask_list=[mask], offload_model=False
                )[0]

                with frame_data:
                    metadata_dict = frame_data.get_frame_metadata(frame_idx)
                    metadata_dict["logo_removed"] = True
                    frame_data.update_frame(index=frame_idx, image=cleaned_frame, metadata_dict=metadata_dict)

                succ_count += 1
            except Exception as e:
                self.logger.info(
                    (
                        f"Was not able to remove logos of frame with index {frame_idx}. "
                        f"Following error occurred:\n{e}"
                    )
                )
        self.logo_removing_model.offload_model()

        self.logger.info((f"Successfully removed logos in {succ_count} frames."))
        return True

    def save_reference_frame(self, frame_data: DynamicVideoArchive) -> bool:
        """Saves the last extracted frame as reference frame.

        Args:
            frame_data: The DynamicVideoArchive for the frame data.

        Returns:
            A bool indicating if the saving was successfull.
        """
        with frame_data:
            if len(frame_data.reference_frames_dset) > 0:
                return True

            if len(frame_data) == 0:
                self.logger.info("Tried to save reference frame but no extracted_frames were found.")
                return False

            frame = frame_data.get_frame(-1)
            metadata = frame_data.get_frame_metadata(-1)
            frame_data.add_reference_frame(frame, metadata)

        self.logger.info("Successfully saved reference frame.")
        return True

    def process(self, video_dir_list: List[str], batch_size: int = -1) -> List[bool]:
        """Extracts frames of a painting tutorial video.

        The processor downloads the video, detects start and end frame as also the canvas.
        Extracts n frames by removing occlusions such as the painters hand and logos.

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
        remove_logos = self.params.get("remove_logos", True)
        compute_median_frame = self.params.get("overwrite_with_median_frame", True)
        num_frames = 1000
        for i, vd in enumerate(video_dir_list):
            video_dir = Path(vd)
            log_file = str(video_dir / "SAM3_Matting.log")
            logging.basicConfig(
                filename=log_file,
                filemode="w",
                force=True,
            )
            video_file_path = str(video_dir / self.video_file_name)
            frame_dataset = DynamicVideoArchive(str(video_dir / self.frame_data))

            # Downloading the video
            if not self._download_video(video_file_path=video_file_path, frame_data=frame_dataset):
                continue

            # Detecting start and end frame
            if not self._detect_start_end_frame(
                video_file_path=video_file_path, frame_data=frame_dataset, batch_size=batch_size
            ):
                continue

            # Detect the canvas if specified
            if detect_canvas and not self.detect_canvas(
                video_dir=video_dir, video_path=video_file_path, frame_data=frame_dataset
            ):
                continue

            # Extracting frames
            if not self.extract_n_frames(
                video_path=video_file_path,
                frame_data=frame_dataset,
                num_frames=num_frames,
                disable_tqdm=disable_tqdm,
            ):
                continue

            if compute_median_frame and not self.overwrite_with_median_frame(frame_data=frame_dataset):
                continue

            # Removes logo and other text from extracted frames
            if remove_logos and not self._remove_logo(frame_data=frame_dataset, disable_tqdm=disable_tqdm):
                continue

            # Save the reference frame
            if not self.save_reference_frame(frame_data=frame_dataset):
                continue

            # Post processing and cleaning up
            if not self._post_process(video_file_path=video_file_path):
                continue

            ret[i] = True

        # Clear file logging
        logging.basicConfig(
            filename=None,
            force=True,
        )

        return ret

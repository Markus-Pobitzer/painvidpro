"""Class for the Loomis Keyframe detection."""
import torch
import logging
import os
from os.path import isfile, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

from painvidpro.logging.logging import setup_logger
from painvidpro.logo_masking.factory import LogoMaskingFactory
from painvidpro.object_detection.base import ObjectDetectionBase
from painvidpro.object_detection.factory import ObjectDetectionBase, ObjectDetectionFactory
from painvidpro.occlusion_masking.factory import OcclusionMaskingBase, OcclusionMaskingFactory
from painvidpro.occlusion_removing.factory import OcclusionRemovingBase, OcclusionRemovingFactory
from painvidpro.processors.keyframe import ProcessorKeyframe
from painvidpro.utils.metadata import load_metadata, save_metadata
from painvidpro.video_processing.utils import video_capture_context, video_writer_context
from painvidpro.video_processing.youtube import download_video
from painvidpro.utils.image_processing import process_input


class SAM3(ProcessorKeyframe):
    def __init__(self):
        """Class to process videos."""
        super().__init__()
        self.set_default_parameters()
        self.logger = setup_logger(name=__name__)
        self._sam3_model: Optional[Sam3Model] = None
        self._sam3_processor: Optional[Sam3Processor] = None
        self.video_file_name = "video.mp4"
        self.metadata_name = "metadata.json"
        self.reference_frame_name = "reference_frame.png"
        self.extr_folder_name = "extracted_frames"
        self.zfill_num = 8
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
        super().set_default_parameters()
        self.params = {
            "yt_video_format": "bestvideo[height<=480]",
            "sam3_model": "facebook/sam3",
            "sam3_config": {},
            "occlusion_masking_config": {
                "prompt": ["a hand", "a paintbrush"]
            },
            "disable_tqdm": True,
            "remove_videos_after_processing": False,
            "start_end_frame_detector_config": {"prompt": "a hand"},
            "device": "cuda"
        }
        self.params["detect_canvas"] = True
        self.params["canvas_detector_config"] = {"prompt": "a blank canvas"}
        self.params["num_bins"] = -1
        self.params["num_samples_per_bin"] = -1
        self.params["remove_logos"] = False
        self.params["logo_masking_config"] = {"prompt": "a logo"}
        self.params["logo_removing_algorithm"] = "OcclusionRemovingLamaInpainting"
        self.params["logo_removing_config"] = {}
        self.params["detect_keyframes"] = False

    def _download_video(self, video_dir: Path, video_file_path: str, metadata: Dict[str, Any]) -> bool:
        """Downloads the video if not alredy downloaded.

        Args:
            video_dir: Path to the current video dir.
            video_file_path: Path to the video file on disk.
            metadata: Metadata as a dict.

        Returns:
            bool: True if successfull or already exists, False otherwise.
        """
        try:
            if not os.path.isfile(video_file_path):
                # TODO: Check which site it is from needs to be done dynamically
                if "youtube" in video_file_path:
                    url = metadata["id"]
                    video_format = self.params.get("yt_video_format", "bestvideo[height<=360]")
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
                            f"The metadata: {metadata}."
                        )
                    )
                    return False

                # Reset that canvas was detected when the video gets downloaded.
                if metadata.get("canvas_detected", False):
                    metadata["canvas_detected"] = False
                    save_metadata(video_dir, metadata_name=self.metadata_name, metadata=metadata)
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
        video_dir: Path,
        video_file_path: str,
        metadata: Dict[str, Any],
        batch_size: int = -1,
        max_frames_to_consider: int = 400,
        frame_steps: int = 30,
    ) -> bool:
        """Detect start and end frame in the video if not already detected.

        Args:
            video_dir: Path to the current video dir.
            video_file_path: Path to the video on disk.
            metadata: Metadata as a dict.
            batch_size: If > 0, then set the batch size of sequence detector
                for detecting start and end frame.
            max_frames_to_conside: max number of frames to consider from the first
                and last instance.
            frame_steps: Take each frame_steps frame.

        Returns:
            Boolean indicating success.
        """
        if "start_frame_idx" not in metadata or "end_frame_idx" not in metadata:
            try:
                if batch_size > 0:
                    self.sequence_detector.params["batch_size"] = batch_size
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
                    print(f"frames_to_consider_bckwrds: {frames_to_consider_bckwrds}")

                    def _first_appearance(frame_idx_list: List[int]) -> int:
                        for frame_idx in frame_idx_list:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                            res, frame = cap.read()
                            if not res:
                                self.logger.info(
                                    f"Was not able to read frame with index {frame_idx} from {video_file_path}."
                                )
                            frame = process_input(frame, convert_bgr_to_rgb=True)
                            inputs = self.sam3_processor(images=frame, text=prompt, return_tensors="pt").to(self.device)
                            with torch.no_grad():
                                outputs = self.sam3_model(**inputs)
                            results = self.sam3_processor.post_process_instance_segmentation(
                                outputs,
                                threshold=0.5,
                                mask_threshold=0.5,
                                target_sizes=inputs.get("original_sizes").tolist()
                            )[0]
                            if len(results) > 0:
                                return frame_idx
                        return -1

                    start_idx = _first_appearance(frame_idx_list=frames_to_consider)
                    end_idx = _first_appearance(frame_idx_list=frames_to_consider_bckwrds)

                metadata["start_frame_idx"] = start_idx
                metadata["end_frame_idx"] = end_idx

                # Save metadata to disk
                save_metadata(video_dir, metadata_name=self.metadata_name, metadata=metadata)
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

            # Detect the canvas
            frame = process_input(frame, convert_bgr_to_rgb=True)
            prompt = self.params.get("canvas_detector_config", {}).get("prompt", "a blank canvas")
            inputs = self.sam3_processor(images=frame, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.sam3_model(**inputs)
            bbox = self.sam3_processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]['boxes'].cpu().to(float).numpy()

            if len(bbox) > 0:
                # We take the first appearance
                x_min, y_min, x_max, y_max = bbox[0]
            else:
                self.logger.info("Was not able to detect a canvas in the video, using the original video for processing.")
                return True

        cavas_only_video_path = str(video_dir / "canvas_only_video.mp4")
        with video_capture_context(video_path=video_path) as cap:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

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

    def _get_frame_indices(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        num_frames_video: int,
        num_bins: int = 10,
        num_samples_per_bin: int = 10,
    ) -> List[np.ndarray]:
        """Get the frame indices associated to the values.

        Args:
            start_frame_idx: The start frame to consider.
            end_frame_idx: The end frame to consider.
            num_frames_video: Total number of frames in the video.
            num_bins: Number of bins to split the video.
            num_samples_per_bin: The number of sampled frames per bin.

        Returns:
            Returns a List, each entry corresponding to a bin, containing
            a numpy array with the frame indices.
        """
        max_num_frames = min(end_frame_idx - start_frame_idx, num_frames_video - start_frame_idx)
        frames_per_bin = max_num_frames // num_bins
        frame_indices = [
            np.linspace(i * frames_per_bin, (i + 1) * frames_per_bin - 1, num=num_samples_per_bin, dtype=int)
            for i in range(num_bins)
        ]
        # Shift frame indices according to start frame
        frame_indices = [fi_array + start_frame_idx for fi_array in frame_indices]

        return frame_indices

    def _extract_median_frame(
        self,
        frame_array: np.ndarray,
        prev_extr_frame: np.ndarray,
        keyframes: np.ndarray,
        rmbg_model: OcclusionMaskingBase,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """Extracts the median frame on the given inputs.

        Applies a mask to `frame_array` to exclude foreground occlusions
        (using `rmbg_model`), then computes the median frame across
        `frame_array`, `prev_extr_frame`, and `keyframes`.
        Masked regions are set to `np.nan` and ignored during median computation.

        Args:
            frame_array: Sampled frames with shape [N, H, W, 3], where N is the number of frames.
            prev_extr_frame: The previously extracted frame with shape [H, W, 3].
            keyframes: Keyframes with shape [M, H, W, 3], where M is the number of keyframes.
            rmbg_model: The occlusion segmentation model.
            kernel_size: Kernel size for occlusion mask dilation.

        Returns:
            The median frame with shape [H, W, 3], computed while ignoring masked regions.
        """
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_list = rmbg_model.compute_mask_list(frame_array, offload_model=False)

        mask_list = [cv2.dilate(np.uint8(mask), kernel, iterations=1).astype(bool) for mask in mask_list]
        mask_array = np.stack(mask_list)  # Convert list of masks to a single array

        # Apply masks to frame_array in a vectorized way
        frame_array[mask_array] = np.nan

        # Concatenate inputs, handling empty keyframes
        inputs = [frame_array, prev_extr_frame[np.newaxis]]
        if keyframes.size > 0:  # Only add keyframes if they are not empty
            inputs.append(keyframes)
        stacked_frames = np.concatenate(inputs, axis=0)
        median_frame = np.uint8(np.nanmedian(stacked_frames, axis=0))
        return median_frame

    def _estimate_sample_strategy(
        self,
        start_frame_idx: int,
        end_frame_idx: int,
        num_frames_video: int,
        fps: int,
        num_bins: int = -1,
        num_samples_per_bin: int = -1,
        sec_per_bin: int = 10,
    ) -> Tuple[int, int]:
        """Compute the number of bins and samples per bin for video sampling.

        If `num_bins` or `num_samples_per_bin` is negative, they are computed based on the video's duration and FPS.
        Otherwise, the provided values are returned as is.

        Args:
            start_frame_idx: The starting frame index to consider.
            end_frame_idx: The ending frame index to consider.
            num_frames_video: Total number of frames in the video.
            fps: Frames per second of the video.
            num_bins: Number of bins to split the video into.
                If negative, it is computed based on `sec_per_bin`.
            num_samples_per_bin: Number of frames to sample per bin.
                If negative, it is computed as 3 * `sec_per_bin`.
            sec_per_bin: Number of seconds of video each bin should
                capture (used only when `num_bins < 0`).

        Returns:
            A tuple of (num_bins, num_samples_per_bin).
        """
        if num_bins < 0:
            max_num_frames = min(end_frame_idx - start_frame_idx, num_frames_video - start_frame_idx)
            # Each bin captures sec_per_bin seconds of content
            frames_per_bin = fps * sec_per_bin
            # The longer the video, the more bins
            num_bins = int(max_num_frames // frames_per_bin)

            # Ensure num_bins is at least 1 and at most 1000
            num_bins = max(1, min(num_bins, 1000))

        if num_samples_per_bin < 0:
            # Sample 3 frames for each second of video
            num_samples_per_bin = 3 * sec_per_bin

        self.logger.info(f"Selected num_bins: {num_bins}\nSelected num_samples_per_bin: {num_samples_per_bin}")

        return num_bins, num_samples_per_bin

    def extract_median_frames(
        self,
        video_dir: Path,
        video_path: str,
        metadata: Dict[str, Any],
        num_bins: int = -1,
        num_samples_per_bin: int = -1,
        kernel_size: int = 5,
        disable_tqdm: bool = True,
    ) -> bool:
        """Extracts median frames from the video and stores them on disk.

        For each bin in the video, a median frame is computed by sampling frames and applying occlusion masking.
        The extracted frames are saved to disk.

        Does nothing if metadata contains `extracted_frames` field.

        Args:
            video_dir: Path to the directory containing the video.
            video_path: Path to the video file.
            metadata: Metadata dict containing 'start_frame_idx' and 'end_frame_idx'.
            num_bins: Number of bins to split the video into. If negative, it is computed automatically.
            num_samples_per_bin: Number of frames to sample per bin. If negative, it is computed automatically.
            kernel_size: Size of the dilation kernel applied to occlusion masks.
            disable_tqdm: If True, disables the tqdm progress bar.

        Returns:
            A bool indicating success. If False it can be:
            - If `end_frame_idx` is smaller than `start_frame_idx`.
            - If reading a frame from the video fails.
            - If no frame is read for a given frame sequence.
        """
        if "extracted_frames" in metadata:
            return True

        start_frame_idx = metadata["start_frame_idx"]
        end_frame_idx = metadata["end_frame_idx"]
        extr_frame_dir = video_dir / self.extr_folder_name
        extr_frame_dir.mkdir(parents=True, exist_ok=True)
        if end_frame_idx < start_frame_idx:
            self.logger.info(
                f"The end_frame_idx {end_frame_idx} must be bigger than start_frame_idx {start_frame_idx}."
            )
            return False
        selected_keyframe_list = metadata.get("selected_keyframe_list", [])

        try:
            with video_capture_context(video_path=video_path) as cap:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                num_bins, num_samples_per_bin = self._estimate_sample_strategy(
                    start_frame_idx, end_frame_idx, num_frames_video, fps, num_bins, num_samples_per_bin
                )
                frame_indices = self._get_frame_indices(
                    start_frame_idx, end_frame_idx, num_frames_video, num_bins, num_samples_per_bin
                )

                median_frame_paths: List[str] = []
                ret_frame_idx: List[int] = []
                frame_idx = 0
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"Was not able to read first frame from {video_path}.")

                # In case, first sampled frames have occluded parts add white canvas
                prev_extr_frame = np.zeros((height, width, 3), dtype=np.float32) + 255

                for f_idx_list in tqdm(frame_indices, desc="Extracting frames", disable=disable_tqdm):
                    size_0 = len(f_idx_list)
                    frame_array = np.zeros((size_0, height, width, 3), dtype=np.float32)

                    # Keyframe list
                    keyframe_list: List[np.ndarray] = []

                    # Read the sampled frames
                    for i, selected_f_idx in enumerate(f_idx_list):
                        while frame_idx < selected_f_idx:
                            ret, frame = cap.read()
                            if not ret:
                                # Something went wrong
                                raise ValueError(f"Reading frame with index {frame_idx} failed.")
                            frame_idx += 1

                            if frame_idx in selected_keyframe_list:
                                keyframe_list.append(frame)

                        if frame is None:
                            raise ValueError(f"Was not able to get frame with index {selected_f_idx}.")
                        frame_array[i] = frame

                    keyframe_array = np.stack(keyframe_list) if keyframe_list else np.array([])
                    median_frame = self._extract_median_frame(
                        frame_array,
                        prev_extr_frame,
                        keyframes=keyframe_array,
                        rmbg_model=self.rmbg_model,
                        kernel_size=kernel_size,
                    )
                    median_frame_idx = int(f_idx_list[size_0 // 2])
                    median_fram_name = f"frame_{str(median_frame_idx).zfill(self.zfill_num)}.png"
                    median_fram_path = str(extr_frame_dir / median_fram_name)
                    cv2.imwrite(median_fram_path, median_frame)

                    ret_frame_idx.append(median_frame_idx)
                    median_frame_paths.append(join(self.extr_folder_name, median_fram_name))
                    # Use the median frame in the computation of the next median frame
                    prev_extr_frame = median_frame

            # Save extracted frames to metadata
            metadata_addition: List[Dict[str, Any]] = []
            for median_frame_idx, median_frame_path in zip(ret_frame_idx, median_frame_paths):
                metadata_addition.append(
                    {
                        "index": median_frame_idx,
                        "path": median_frame_path,
                        "extraction_method": "median",
                    }
                )
            metadata["extracted_frames"] = metadata.get("extracted_frames", []) + metadata_addition
            save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)

        except Exception as e:
            self.logger.info(f"Was not able to extract median frames from {video_path}: {e}")
            return False

        self.logger.info(f"Successfully extracted {len(median_frame_paths)} median frames.")
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
        prompt = self.params.get("canvas_detector_config", {}).get("prompt", ["a hand", "a paintbrush"])
        if isinstance(prompt, str):
            prompt = [prompt]
        frame_list = [frame] * len(prompt)
        # Could run OOM if to many prompt items
        inputs = self.sam3_processor(images=frame_list, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.sam3_model(**inputs)
        results = self.sam3_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )

        # Extract masks from result. Each index in the list corresponds to a prompt.
        mask_list: List[np.ndarray] = [res['masks'].cpu().numpy().any(axis=0) for res in results]
        # Combine masks of all prompts
        mask = np.stack(mask_list, axis=0).any(axis=0)
        # Dilate for better coverage
        mask = cv2.dilate(np.uint8(mask), kernel, iterations=1).astype(bool)

        if prev_extr_frame is not None:
            ret_frame = np.where(mask, frame, prev_extr_frame)
        else:
            # Notation a bit misleading but here we convert from rgb to bgr (it is just a channel switch).
            bgr_frame = process_input(frame, convert_bgr_to_rgb=True)
            ret_frame = self.logo_removing_model.remove_occlusions(
                frame_list=[bgr_frame], mask_list=[mask], offload_model=False
            )[0]
        return ret_frame

    def extract_n_frames(
        self,
        video_dir: Path,
        video_path: str,
        metadata: Dict[str, Any],
        num_frames: int = 1000,
        kernel_size: int = 5,
        disable_tqdm: bool = True,
    ) -> bool:
        """Extracts N frames from the video and stores them on disk.

        Does nothing if metadata contains `extracted_frames` field.

        Args:
            video_dir: Path to the directory containing the video.
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
        if "extracted_frames" in metadata:
            return True

        start_frame_idx = metadata["start_frame_idx"]
        end_frame_idx = metadata["end_frame_idx"]
        extr_frame_dir = video_dir / self.extr_folder_name
        extr_frame_dir.mkdir(parents=True, exist_ok=True)
        if end_frame_idx < start_frame_idx:
            self.logger.info(
                f"The end_frame_idx {end_frame_idx} must be bigger than start_frame_idx {start_frame_idx}."
            )
            return False

        try:
            with video_capture_context(video_path=video_path) as cap:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                num_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                num_frames = min(num_frames, num_frames_video)

                frame_indices = np.linspace(start=start_frame_idx, stop=end_frame_idx, num=num_frames).tolist()

                frame_paths: List[str] = []
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

                        cleaned_frame = self._extract_frame(frame=frame, prev_extr_frame=prev_frame, kernel_size=kernel_size)

                        fram_name = f"frame_{str(frame_idx).zfill(self.zfill_num)}.png"
                        fram_path = str(extr_frame_dir / fram_name)
                        cv2.imwrite(fram_path, cleaned_frame)
                        frame_paths.append(join(self.extr_folder_name, fram_name))
                        prev_frame = cleaned_frame

            # Save extracted frames to metadata
            metadata_addition: List[Dict[str, Any]] = []
            for median_frame_idx, median_frame_path in zip(frame_indices, frame_paths):
                metadata_addition.append(
                    {
                        "index": median_frame_idx,
                        "path": median_frame_path,
                        "extraction_method": "sam3",
                    }
                )
            metadata["extracted_frames"] = metadata.get("extracted_frames", []) + metadata_addition
            save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)

        except Exception as e:
            self.logger.info(f"Was not able to extract frames from {video_path}: {e}")
            return False

        self.logger.info(f"Successfully extracted {len(frame_paths)} frames.")
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

        Processes all frames listed in metadata's "extracted_frames" by first detecting logos
        with a masking model, then removing them. Overwrites original frames
        with cleaned versions.

        Args:
            video_dir: Directory containing extracted frames from video.
            metadata: Metadata as dict.
            disable_tqdm: If True, disables progress bar visualization.

        Returns:
            bool: Always returns True, indicating completion of processing attempts for all frames.
                  Individual frame failures are logged but don't prevent overall completion.
        """
        extracted_frame_list: List[str] = metadata.get("extracted_frames", [])
        succ_count = 0
        frame_path_list: List[str] = []
        mask_list: List[np.ndarray] = []

        # Detection
        for extr_frame_entry in tqdm(extracted_frame_list, disable=disable_tqdm, desc="Detecting Logos from frames"):
            frame_path = str(video_dir / extr_frame_entry["path"])
            try:
                img = cv2.imread(frame_path)
                mask = self.logo_masking_model.compute_mask_list(frame_list=[img], offload_model=False)[0]
                # Only append when no exception occured
                frame_path_list.append(frame_path)
                mask_list.append(mask)
            except Exception as e:
                self.logger.info(
                    (f"Was not able to detect Logos of frame {frame_path}. " f"Following error occurred:\n{e}")
                )
        self.logo_masking_model.offload_model()

        if len(mask_list) == 0:
            self.logger.info("No frames left to remove logos from.")
            return True

        # Computing the mask where fix logos most likely are
        video_mask = self._create_combined_mask(mask_list=mask_list)
        mask_list = [np.logical_or(mask, video_mask) for mask in mask_list]

        # Removing
        for frame_path, mask in tqdm(
            zip(frame_path_list, mask_list), disable=disable_tqdm, desc="Removing Logos from frames"
        ):
            try:
                img = cv2.imread(frame_path)
                cleaned_list = self.logo_removing_model.remove_occlusions(
                    frame_list=[img], mask_list=[mask], offload_model=False
                )
                # Since the output is in RGB convert to BGR
                cleaned_img = cv2.cvtColor(cleaned_list[0], cv2.COLOR_RGB2BGR)
                # Overwriting frame with cleaned one
                cv2.imwrite(frame_path, cleaned_img)
                succ_count += 1
            except Exception as e:
                self.logger.info(
                    (f"Was not able to remove logos of frame {frame_path}. " f"Following error occurred:\n{e}")
                )
        self.logo_removing_model.offload_model()

        self.logger.info(
            (
                f"Successfully removed logos in {succ_count} frames. "
                f"In total there are {len(extracted_frame_list)} frames."
            )
        )
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
        num_bins = self.params.get("num_bins", -1)
        num_samples_per_bin = self.params.get("num_samples_per_bin", -1)
        detect_canvas = self.params.get("detect_canvas", True)
        remove_logos = self.params.get("remove_logos", False)
        detect_keyframes = self.params.get("detect_keyframes", False)
        for i, vd in enumerate(video_dir_list):
            video_dir = Path(vd)
            log_file = str(video_dir / "ProcessorMatting.log")
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
            if not self._download_video(video_dir=video_dir, video_file_path=video_file_path, metadata=metadata):
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

            # Detecting and extracting the keyframes if specified
            if detect_keyframes and not self._detect_keyframes(
                video_dir=video_dir, video_file_path=video_file_path, metadata=metadata
            ):
                continue

            # Extracting frames by computing the median of sampled frames
            if not self.extract_median_frames(
                video_dir=video_dir,
                video_path=video_file_path,
                metadata=metadata,
                num_bins=num_bins,
                num_samples_per_bin=num_samples_per_bin,
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

            # Offloading to cpu
            try:
                self.rmbg_model.model.to("cpu")  # type: ignore
            except Exception as e:
                self.logger.info(f"Was not able to offload the RMBG model: {e}")

        # Clear file logging
        logging.basicConfig(
            filename=None,
            force=True,
        )

        return ret

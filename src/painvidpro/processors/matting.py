"""Class for the Loomis Keyframe detection."""

import logging
import os
from os.path import isfile, join
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from painvidpro.logo_masking.factory import LogoMaskingBase, LogoMaskingFactory
from painvidpro.object_detection.factory import ObjectDetectionBase, ObjectDetectionFactory
from painvidpro.occlusion_masking.factory import OcclusionMaskingBase, OcclusionMaskingFactory
from painvidpro.occlusion_removing.factory import OcclusionRemovingBase, OcclusionRemovingFactory
from painvidpro.processors.keyframe import ProcessorKeyframe
from painvidpro.utils.metadata import load_metadata, save_metadata
from painvidpro.video_processing.utils import video_capture_context, video_writer_context


class ProcessorMatting(ProcessorKeyframe):
    def __init__(self):
        """Class to process videos."""
        super().__init__()
        self._canvas_detector: Optional[ObjectDetectionBase] = None
        self._rmbg_model: Optional[OcclusionMaskingBase] = None
        self._logo_masking_model: Optional[LogoMaskingBase] = None
        self._logo_removing_model: Optional[OcclusionRemovingBase] = None

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
    def rmbg_model(self) -> OcclusionMaskingBase:
        if self._rmbg_model is None:
            raise RuntimeError(
                (
                    "Occlusion Masking model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model and processor."
                )
            )
        return self._rmbg_model

    @property
    def logo_masking_model(self) -> LogoMaskingBase:
        if self._logo_masking_model is None:
            raise RuntimeError(
                (
                    "Logo Masking model not correctly instanciated. Make sure to call "
                    "set_parameters to laod the model."
                )
            )
        return self._logo_masking_model

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
        ret, msg = super().set_parameters(params)
        if not ret:
            return ret, msg
        occlusion_masking_algorithm = self.params.get("occlusion_masking_algorithm", "OcclusionMaskingInSPyReNet")
        occlusion_masking_config = self.params.get("occlusion_masking_config", {})
        detect_canvas = self.params.get("detect_canvas", False)
        remove_logos = self.params.get("remove_logos", False)
        try:
            if detect_canvas:
                self._canvas_detector = ObjectDetectionFactory().build(
                    self.params["canvas_detector_algorithm"], self.params["canvas_detector_config"]
                )
            self._rmbg_model = OcclusionMaskingFactory().build(occlusion_masking_algorithm, occlusion_masking_config)
            if remove_logos:
                self._logo_masking_model = LogoMaskingFactory().build(
                    self.params["logo_masking_algorithm"], self.params["logo_masking_config"]
                )
                self._logo_removing_model = OcclusionRemovingFactory().build(
                    self.params["logo_removing_algorithm"], self.params["logo_removing_config"]
                )
        except Exception as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        super().set_default_parameters()
        self.params["detect_canvas"] = False
        self.params["canvas_detector_algorithm"] = "ObjectDetectionGroundingDino"
        self.params["canvas_detector_config"] = {"prompt": "a canvas."}
        self.params["occlusion_masking_algorithm"] = "OcclusionMaskingInSPyReNet"
        self.params["occlusion_masking_config"] = {}
        self.params["num_bins"] = -1
        self.params["num_samples_per_bin"] = -1
        self.params["remove_logos"] = False
        self.params["logo_masking_algorithm"] = "LogoMaskingGroundingDino"
        self.params["logo_masking_config"] = {}
        self.params["logo_removing_algorithm"] = "OcclusionRemovingLamaInpainting"
        self.params["logo_removing_config"] = {}
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
        for rel_path in tqdm(extracted_frame_list, disable=disable_tqdm, desc="Removing Logos from frames"):
            frame_path = str(video_dir / rel_path)
            try:
                img = cv2.imread(frame_path)
                mask_list = self.logo_masking_model.compute_mask_list(frame_list=[img], offload_model=False)
                cleaned_list = self.logo_removing_model.remove_occlusions(
                    frame_list=[img], mask_list=mask_list, offload_model=False
                )
                # Since the output is in RGB convert to BGR
                cleaned_img = cv2.cvtColor(cleaned_list[0], cv2.COLOR_RGB2BGR)
                # Overwriting frame with cleaned one
                cv2.imwrite(frame_path, cleaned_img)
                succ_count += 1
            except Exception as e:
                self.logger.info(
                    (f"Was not able to remove Logos of frame {frame_path}. " f"Following error occurred:\n{e}")
                )
        self.logo_masking_model.offload_model()
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

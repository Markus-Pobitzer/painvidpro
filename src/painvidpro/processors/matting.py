"""Class for the Loomis Keyframe detection."""

import logging
from os.path import isfile, join
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from painvidpro.occlusion_masking.factory import OcclusionMaskingFactory, OcclusionMaskingRMBG
from painvidpro.processors.keyframe import ProcessorKeyframe
from painvidpro.utils.metadata import load_metadata, save_metadata
from painvidpro.video_processing.utils import video_capture_context


class ProcessorMatting(ProcessorKeyframe):
    def __init__(self):
        """Class to process videos."""
        super().__init__()

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
        occlusion_masking_algorithm = self.params.get("occlusion_masking_algorithm", "OcclusionMaskingRMBG")
        occlusion_masking_config = self.params.get("occlusion_masking_config", {})
        self.rmbg_model: OcclusionMaskingRMBG = OcclusionMaskingFactory().build(
            occlusion_masking_algorithm, occlusion_masking_config
        )  # type: ignore
        return True, ""

    def set_default_parameters(self):
        super().set_default_parameters()
        self.params["occlusion_masking_algorithm"] = "OcclusionMaskingRMBG"
        self.params["occlusion_masking_config"] = {}
        self.params["num_bins"] = -1
        self.params["num_samples_per_bin"] = -1

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
        rmbg_model: OcclusionMaskingRMBG,
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

    def save_reference_frame(self, metadata: Dict[str, Any], reference_frame_path: str) -> bool:
        """Saves the last extracted frame as reference frame.

        Args:
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
            cv2.imwrite(reference_frame_path, cv2.imread(selected_frame["path"]))
        except Exception as e:
            self.logger.info(
                (
                    f"Tried to save reference frame but was not able to load image from {selected_frame.get('path', "no path was found")}."
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

            # Detecting and extracting the keyframes
            if not self._detect_keyframes(video_dir=video_dir, video_file_path=video_file_path, metadata=metadata):
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

            # Save the reference frame
            if not self.save_reference_frame(metadata=metadata, reference_frame_path=reference_frame_path):
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

"""Class for the Loomis Keyframe detection."""

import logging
import os
from os.path import isfile
from pathlib import Path
from typing import Any, Dict, List

import cv2

from painvidpro.processors.matting import ProcessorMatting
from painvidpro.utils.metadata import load_metadata, save_metadata
from painvidpro.video_processing.split import estimate_split_width_from_frame, split_horizontal
from painvidpro.video_processing.utils import median_of_video


class ProcessorLoomisMatting(ProcessorMatting):
    def __init__(self):
        """Class to process videos."""
        super().__init__()
        self.video_portrait_name = "video_left.mp4"
        self.video_painting_name = "video_right.mp4"

    def set_default_parameters(self):
        super().set_default_parameters()
        # Overwriting youtube video format
        self.params["yt_video_format"] = "bestvideo[height<=480]"

    def extract_reference_frame(self, video_file_path: str, reference_frame_path: str) -> bool:
        """Extract the reference frame from a video and save it.

        It samples frames from the video and computes the median of it.
        Only extracts and saves it if not already present.

        Args:
            video_file_path: The path to the video.
            reference_frame_path: The reference frame path to save the extracted image.

        Returns:
            A bool indicating if it was successfull.
        """
        if isfile(reference_frame_path):
            return True

        try:
            # Extracting the portrait image
            median_frame = median_of_video(video_file_path)
            median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(reference_frame_path, median_frame)
        except Exception as e:
            self.logger.info(
                (f" Failed to extract reference image from video {video_file_path}." f"\nException:\n{e}")
            )
            return False
        return True

    def _split_video(self, video_file_path: str, video_left: str, video_right: str, metadata: Dict[str, Any]) -> bool:
        """Splits the video_file_path video into two parts.

        First split width is estimated by checking at which
        vertical line the pixel change is greatest.

        Args:
            video_file_path: The path to the video.
            video_left: The path where the left side of the video gets stored.
            video_right: The path where the right side of the video gets stored.
            metadata: The metadata dict.

        Returns:
            A bool indicating if it was successfull.
        """
        if os.path.isfile(video_left) and os.path.isfile(video_right):
            return True

        frame_index = metadata["start_frame_idx"]
        try:
            # Estimating the split width
            split_width = estimate_split_width_from_frame(
                video_file_path=video_file_path, frame_index=frame_index, left_border=0.4, right_border=0.6
            )
        except Exception as e:
            self.logger.info(
                (
                    f" Failed to estimate split_width from video {video_file_path} "
                    f"with frame_index = {frame_index}:\nException: {e}."
                )
            )
            return False

        try:
            # Splitting the real image: portrait of model left and painting video on the right
            split_horizontal(video_file_path, video_left, video_right, split_width)
        except Exception as e:
            self.logger.info((f" Failed to split video {video_file_path}." f"\nException:\n{e}"))
            return False
        return True

    def _post_process_loomis(self, video_file_path: str, video_left: str, video_right: str) -> bool:
        """Does post processing.

        Includes deletion of files if specified.

        Args:
            video_file_path: Path to the video on disk.

        Returns:
            A bool indicating success.
        """
        if self.params.get("remove_videos_after_processing"):
            self._delete_video(video_file_path=video_file_path)
            self._delete_video(video_file_path=video_left)
            self._delete_video(video_file_path=video_right)
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
        detect_keyframes = self.params.get("detect_keyframes", False)
        for i, vd in enumerate(video_dir_list):
            video_dir = Path(vd)
            log_file = str(video_dir / "ProcessorLoomisMatting.log")
            logging.basicConfig(
                filename=log_file,
                filemode="w",
                force=True,
            )
            video_file_path = str(video_dir / self.video_file_name)
            video_left = str(video_dir / self.video_portrait_name)
            video_right = str(video_dir / self.video_painting_name)
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

            # Split the video
            if not self._split_video(
                video_file_path=video_file_path, video_left=video_left, video_right=video_right, metadata=metadata
            ):
                continue

            # Save the reference frame
            if not self.extract_reference_frame(video_file_path=video_left, reference_frame_path=reference_frame_path):
                continue

            # Detecting and extracting the keyframes
            if detect_keyframes and not self._detect_keyframes(
                video_dir=video_dir, video_file_path=video_right, metadata=metadata
            ):
                continue

            # Extracting frames by computing the median of sampled frames
            if not self.extract_median_frames(
                video_dir=video_dir,
                video_path=video_right,
                metadata=metadata,
                num_bins=num_bins,
                num_samples_per_bin=num_samples_per_bin,
                disable_tqdm=disable_tqdm,
            ):
                continue

            # Post processing and cleaning up
            if not self._post_process_loomis(
                video_file_path=video_file_path, video_left=video_left, video_right=video_right
            ):
                continue

            # Processing was successfull
            metadata["processed_video_name"] = self.video_painting_name
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

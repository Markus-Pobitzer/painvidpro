"""Class for the Loomis Keyframe detection."""

import logging
import os
from pathlib import Path
from typing import List

import cv2

from painvidpro.processors.keyframe import ProcessorKeyframe
from painvidpro.utils.metadata import load_metadata, save_metadata
from painvidpro.video_processing.split import estimate_split_width_from_frame, split_horizontal
from painvidpro.video_processing.utils import median_of_video


class ProcessorLoomisKeyframe(ProcessorKeyframe):
    def __init__(self):
        """Class to process videos."""
        super().__init__()
        self.video_portrait_name = "video_left.mp4"
        self.video_painting_name = "video_right.mp4"

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
        for i, vd in enumerate(video_dir_list):
            video_dir = Path(vd)
            log_file = str(video_dir / "ProcessorLoomisKeyframe.log")
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

            # Estmating the split width
            frame_index = metadata["start_frame_idx"]
            try:
                split_width = estimate_split_width_from_frame(
                    video_file_path=video_file_path, frame_index=frame_index, left_border=0.4, right_border=0.6
                )
            except Exception as e:
                self.logger.info(
                    (
                        f" Failed to estimate split_width from video {video_file_path} "
                        f"with frame_index = {frame_index}: {e}."
                    )
                )
                continue

            if not os.path.isfile(video_left) or not os.path.isfile(video_right):
                try:
                    # Splitting the real image: portrait of model left and painting video on the right
                    split_horizontal(video_file_path, video_left, video_right, split_width)
                except Exception as e:
                    self.logger.info((f" Failed to split video {video_file_path}." f"\nException:\n{e}"))
                    continue

            if not os.path.isfile(reference_frame_path):
                try:
                    # Extracting the portrait image
                    median_frame = median_of_video(video_left)
                    median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(reference_frame_path, median_frame)
                except Exception as e:
                    self.logger.info(
                        (f" Failed to extract reference image from video {video_left}." f"\nException:\n{e}")
                    )
                    continue

            # Detecting and extracting the keyframes
            if not self._detect_keyframes(video_dir=video_dir, video_file_path=video_right, metadata=metadata):
                continue

            # Processing was successfull
            metadata["processed_video_name"] = self.video_painting_name
            metadata["processed"] = True
            save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)
            ret[i] = True

        # Clear file logging
        logging.basicConfig(
            filename=None,
            force=True,
        )
        return ret

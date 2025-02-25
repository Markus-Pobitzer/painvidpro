"""Class for the Keyframe detection."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from painvidpro.keyframe_detection.base import KeyframeDetectionBase
from painvidpro.keyframe_detection.factory import KeyframeDetectionFactory
from painvidpro.processors.base import ProcessorBase
from painvidpro.sequence_detection.base import SequenceDetectionBase
from painvidpro.sequence_detection.factory import SequenceDetectionFactory
from painvidpro.utils.metadata import load_metadata, save_metadata
from painvidpro.video_processing.utils import video_capture_context
from painvidpro.video_processing.youtube import download_video


class ProcessorKeyframe(ProcessorBase):
    def __init__(self):
        """Class to process videos."""
        super().__init__()
        self.set_default_parameters()
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
            datefmt="%d-%m-%Y %H:%M:%S",
        )
        self.sequence_detector: SequenceDetectionBase
        self.keyframe_detector: KeyframeDetectionBase
        self.video_file_name = "video.mp4"
        self.metadata_name = "metadata.json"
        self.keyframe_folder_name = "keyframes"
        self.reference_frame_name = "reference_frame.png"
        self.extr_folder_name = "extracted_frames"
        self.zfill_num = 8

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        self.params.update(params)
        try:
            self.sequence_detector = SequenceDetectionFactory().build(
                self.params["sequence_detection_algorithm"], self.params["sequence_detection_config"]
            )
            self.keyframe_detector = KeyframeDetectionFactory().build(
                self.params["keyframe_detection_algorithm"], self.params["keyframe_detection_config"]
            )
        except ValueError as e:
            return False, str(e)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "sequence_detection_algorithm": "SequenceDetectionGroundingDino",
            "sequence_detection_config": {},
            "keyframe_detection_algorithm": "KeyframeDetectionFrameDiff",
            "keyframe_detection_config": {},
            "disable_tqdm": True,
        }

    def _download_video(self, video_file_path: str, metadata: Dict[str, Any]) -> bool:
        """Downloads the video if not alredy downloaded.

        Args:
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
                download_video(url, video_file_path)
        except Exception as e:
            self.logger.info(f" Failed downloading {video_file_path}: {e}")
            return False
        return True

    def _detect_start_end_frame(
        self, video_dir: Path, video_file_path: str, metadata: Dict[str, Any], batch_size: int = -1
    ) -> bool:
        """Detect start and end frame in the video if not already detected.

        Args:
            video_file_path: Path to the video on disk.
            metadata: Metadata as a dict.
            batch_size: If > 0, then set the batch size of sequence detector
                for detecting start and end frame.

        Returns:
            Boolean indicating success.
        """
        if "start_frame_idx" not in metadata or "end_frame_idx" not in metadata:
            try:
                if batch_size > 0:
                    self.sequence_detector.params["batch_size"] = batch_size
                # Detecting start and end frame
                sequence_list = self.sequence_detector.detect_sequences_on_disk(frame_path=video_file_path)
                metadata["start_frame_idx"] = sequence_list[0].start_idx
                metadata["end_frame_idx"] = sequence_list[0].end_idx

                # Save metadata to disk
                save_metadata(video_dir, metadata_name=self.metadata_name, metadata=metadata)
            except Exception as e:
                self.logger.info(f" Failed detecting start and end frame from {video_file_path}: {e}")
                return False

        return True

    def _detect_keyframes(self, video_dir: Path, video_file_path: str, metadata: Dict[str, Any]) -> bool:
        """Detects the keyframes if not already done.

        Args:
            video_dir: Path to the folder containing the video.
            video_file_path: Path to the video file.
            metadata: Metadata as a dict.

        Returns:
            Boolean indicating success.
        """
        if "keyframe_list" not in metadata:
            try:
                start_frame_idx = metadata["start_frame_idx"]
                end_frame_idx = metadata["end_frame_idx"]

                # Detect keyframes
                keyframe_list = self.keyframe_detector.detect_keyframes_on_disk(frame_path=video_file_path)
                keyframe_dir = video_dir / self.keyframe_folder_name
                keyframe_dir.mkdir(parents=True, exist_ok=True)
                # Write them to disk
                selected_keyframe_list: List[int] = []
                for keyframes in keyframe_list:
                    # Only take keyframes that are between start and end frame
                    if keyframes[0] >= start_frame_idx and keyframes[-1] <= end_frame_idx:
                        keyframe_idx = keyframes[len(keyframes) // 2]
                        selected_keyframe_list.append(keyframe_idx)
                # Should be sorted, just to make sure
                selected_keyframe_list.sort()

                if len(selected_keyframe_list) > 0:
                    # We explicetly iterate over all frames to be sure we get the one
                    # see: https://github.com/opencv/opencv/issues/9053
                    with video_capture_context(video_path=video_file_path) as cap:
                        j = 0
                        for i in range(end_frame_idx + 1):
                            res, frame = cap.read()
                            if not res:
                                break
                            if i == selected_keyframe_list[j]:
                                keyframe_path = f"frame_{str(i).zfill(self.zfill_num)}.png"
                                keyframe_path = str(keyframe_dir / keyframe_path)
                                cv2.imwrite(keyframe_path, frame)
                                j += 1
                                if j >= len(selected_keyframe_list):
                                    break

                metadata["keyframe_list"] = keyframe_list
                metadata["selected_keyframe_list"] = selected_keyframe_list
                save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)
            except Exception as e:
                self.logger.info(f" Failed detecting keyframes for {video_file_path}: {e}")
                return False
        return True

    def process(self, video_dir_list: List[str], batch_size: int = -1) -> List[bool]:
        """Extracts the Keyframes of the videos.

        The processor downloads the video, detects the start/end frame
        in the video and extracts keyframes between start/end frame.

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
            video_file_path = str(video_dir / self.video_file_name)

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

            if not self._detect_keyframes(video_dir=video_dir, video_file_path=video_file_path, metadata=metadata):
                continue

            # Processing was successfull
            metadata["processed_video_name"] = self.video_file_name
            metadata["processed"] = True
            save_metadata(video_dir=video_dir, metadata=metadata, metadata_name=self.metadata_name)
            ret[i] = True
        return ret

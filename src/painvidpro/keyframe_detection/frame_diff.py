"""Base class for the Keyframe detection."""

from typing import List, Union

import cv2
import numpy as np
from tqdm import tqdm

from painvidpro.keyframe_detection.base import KeyframeDetectionBase
from painvidpro.utils.image_processing import process_input
from painvidpro.video_processing.utils import video_capture_context


class KeyframeDetectionFrameDiff(KeyframeDetectionBase):
    def __init__(self):
        """Class to detect keyframes."""
        super().__init__()
        self.set_default_parameters()

    def set_default_parameters(self):
        self.params = {
            "diff_threshold": 10,
            "max_num_changes": 50,
            "disable_tqdm": True,
        }

    def _detect_keyframes(
        self,
        frame_list: Union[Union[List[np.ndarray], List[str]], cv2.VideoCapture],
        numb_frames: int,
        diff_threshold: int = 10,
        max_num_changes: int = 50,
        disable_tqdm: bool = True,
    ) -> List[List[int]]:
        """
            Detects frames which have no occlusions.

            Original code: https://github.com/CraGL/timelapse/blob/master\
                /1%20preprocessing/s2_extract_keyframes/detect_keyframe.cpp

            Args:
                frame_list: List of frames.
                numb_frames: The number of frames.
                diff_threshold: Threshold for the difference in pixels.
                max_num_changes: The maximum number of pixels that can be
                    changed such that no movments are picked up in the frame.
                disable_tqdm: If set disables the progressbar.

            Returns:
                List of frame sequences. Each sequence corresponds to one Keyframe.
                Note that several frames can correspond to one Keyframe if there
                is no changes between them.

            Raises:
                TypeError: If the content of frame_list is neither an np.ndarray nor a string.
                ValueError: If the string does not correspond to a valid image path.
        """
        N = numb_frames

        # For the sign array we have following:
        # 0 means less than max_num_changes pixels changed
        # 1 means more than max_num_changes pixels changed
        sign_array = [0] * N

        if isinstance(frame_list, List):
            frame = process_input(frame_list[0])
        else:
            frame = process_input(frame_list)
        img_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

        # Compute the difference from one frame to the next
        for i in tqdm(range(1, N), desc="Keyframe detection [compute frame diff]", disable=disable_tqdm):
            if isinstance(frame_list, List):
                frame = process_input(frame_list[i])
            else:
                frame = process_input(frame_list)
            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

            diff = np.sqrt(np.sum((frame_lab - img_lab) ** 2, axis=-1))
            count_changes = (diff > diff_threshold).sum()

            if count_changes >= max_num_changes:
                sign_array[i] = 1

            img_lab = frame_lab.copy()

        # selected_sign[i] == 0, means nothing special.
        # selected_sign[i] == 1, no change in previous, current, and next frame
        #                        but incoming change at frame i + 2.
        # selected_sign[i] == 2, previous frame changed but current and next two
        #                        do not.
        selected_sign = [0] * N
        selected_sign[0] = 2
        selected_sign[N - 1] = 1

        for i in range(2, N - 2):
            if sign_array[i - 1 : i + 3] == [0, 0, 0, 1]:
                selected_sign[i] = 1
            if sign_array[i - 1 : i + 3] == [1, 0, 0, 0]:
                selected_sign[i] = 2

        flag = 0
        ret: List[List[int]] = []
        keyframe_sequence: List[int] = []

        last_positon = N
        # Index to start checking for possible keyframes
        for i in range(N - 1, -1, -1):
            if selected_sign[i] == 1:
                last_positon = i
                break

        i = 0
        # If we think we have a sequence of unmoving frames, save the average as a keyframe.
        # Save the frames in between to apply color shifts.
        while i < N:
            # Start of a frame sequence that is good
            if selected_sign[i] == 2:
                keyframe_sequence = [i]
                flag = 2

            # Process frames in the good sequence until a change is noticeable
            if flag == 2:
                i += 1
                keyframe_sequence.append(i)

                # This frame is still good but a frame with changes incoming
                if selected_sign[i] == 1:
                    ret.append(keyframe_sequence)
                    # Indicating that the good sequence ends
                    flag = 1

            if flag == 1:
                if i == last_positon:
                    break
                i += 1

        return ret

    def detect_keyframes(self, frame_list: Union[List[np.ndarray], List[str]]) -> List[List[int]]:
        """
        Detects frames which have no occlusions.

        Args:
            frame_list: List of frames in cv2 image format or path on disk if string.

        Returns:
            List of frame sequences. Each sequence corresponds to one Keyframe.
            Note that several frames can correspond to one Keyframe if there
            is no changes between them.

        Raises:
            TypeError: If the content of frame_list is not an np.ndarray.
        """
        diff_threshold = self.params.get("diff_threshold", 10)
        max_num_changes = self.params.get("max_num_changes", 50)
        disable_tqdm = self.params.get("disable_tqdm", True)
        return self._detect_keyframes(
            frame_list=frame_list,
            numb_frames=len(frame_list),
            diff_threshold=diff_threshold,
            max_num_changes=max_num_changes,
            disable_tqdm=disable_tqdm,
        )

    def detect_keyframes_on_disk(self, frame_path: str) -> List[List[int]]:
        """
        Detects frames which have no occlusions.

        Images are loaded directly from disk

        Args:
            frame_path: Video file path.

        Returns:
            List of frame sequences. Each sequence corresponds to one Keyframe.
            Note that several frames can correspond to one Keyframe if there
            is no changes between them.

        Raises:
            TypeError: If the content of frame_path is not a string.
            ValueError: If the string does not correspond to a valid image path.
        """
        diff_threshold = self.params.get("diff_threshold", 10)
        max_num_changes = self.params.get("max_num_changes", 50)
        disable_tqdm = self.params.get("disable_tqdm", True)
        with video_capture_context(video_path=frame_path) as cap:
            numb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return self._detect_keyframes(
                frame_list=cap,
                numb_frames=numb_frames,
                diff_threshold=diff_threshold,
                max_num_changes=max_num_changes,
                disable_tqdm=disable_tqdm,
            )

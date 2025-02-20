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
            "frame_step": 1,
            "num_still_frames": 3,
            "disable_tqdm": True,
        }

    def _detect_still_frames_timelapse(self, sign_array: List[int]) -> List[List[int]]:
        """Detect keyframes as still frames.

        Original code: https://github.com/CraGL/timelapse/blob/master\
                /1%20preprocessing/s2_extract_keyframes/detect_keyframe.cpp

        Args:
            sign_array: The sign array.

        Returns:
            The keyframes as a list of list.
        """
        # selected_sign[i] == 0, means nothing special.
        # selected_sign[i] == 1, no change in previous, current, and next frame
        #                        but incoming change at frame i + 2.
        # selected_sign[i] == 2, previous frame changed but current and next two
        #                        do not.
        N = len(sign_array)
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

    def _detect_still_frames(self, sign_array: List[int], num_still_frames: int = 3) -> List[List[int]]:
        """Detect keyframes as still frames.

        Handles beginning and end of sign array slightly different then
        _detect_still_frames_timelapse.

        Args:
            sign_array: The sign array.
            num_still_frames: The required number of frames without movements such
                that it gets detected as a keyframe sequence.

        Returns:
            The keyframes as a list of list.
        """
        ret: List[List[int]] = []

        if len(sign_array) < (num_still_frames + 2):
            return ret

        keyframe_sequence: List[int] = [0]
        # A keyframe sequence starts with [1, 0, 0, 0, ...]
        sequence_start = [1] + [0] * num_still_frames
        # A keyframe sequence ends with [0, 0, 0, ..., 1]
        sequence_end = [0] * num_still_frames + [1]
        nsf_dec = num_still_frames - 1
        for i in range(1, len(sign_array) - nsf_dec):
            if sign_array[i - 1 : i + num_still_frames] == sequence_start:
                # [1, j, 0, 0, ...] with j == sign_array[i] == 0
                keyframe_sequence = [i]
            elif sign_array[i - 1 : i + num_still_frames] == sequence_end:
                # [0, j, 0, ..., 1] with j == sign_array[i] == 0
                keyframe_sequence.append(i)
                ret.append(keyframe_sequence)
                keyframe_sequence = []
            elif len(keyframe_sequence) > 0:
                if sign_array[i] == 0:
                    keyframe_sequence.append(i)
                else:
                    keyframe_sequence = []

        if len(keyframe_sequence) > 0 and sign_array[-nsf_dec] == 0:
            # In case we have an open sequence
            n = len(sign_array)
            keyframe_sequence += [n - nsf_dec]
            ret.append(keyframe_sequence)

        return ret

    def _detect_keyframes(
        self,
        frame_list: Union[Union[List[np.ndarray], List[str]], cv2.VideoCapture],
        numb_frames: int,
        diff_threshold: int = 10,
        max_num_changes: int = 50,
        frame_step: int = 1,
        num_still_frames: int = 3,
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
                frame_step: An integer specifiying the frame incrementation.
                num_still_frames: The required number of frames without movements such
                    that it gets detected as a keyframe sequence.
                disable_tqdm: If set disables the progressbar.

            Returns:
                List of frame sequences. Each sequence corresponds to one Keyframe.
                Note that several frames can correspond to one Keyframe if there
                is no changes between them.

            Raises:
                TypeError: If the content of frame_list is neither an np.ndarray nor a string.
                ValueError: If the string does not correspond to a valid image path.
        """
        if frame_step < 0:
            frame_step = 1
        N = (numb_frames - 1) // frame_step + 1

        assert len(range(0, numb_frames, frame_step)) == N

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

            if i % frame_step != 0:
                # Skip this frame
                continue

            frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            diff = np.sqrt(np.sum((frame_lab - img_lab) ** 2, axis=-1))
            count_changes = (diff > diff_threshold).sum()

            if count_changes >= max_num_changes:
                sign_array[i // frame_step] = 1

            img_lab = frame_lab

        still_frames = self._detect_still_frames(sign_array=sign_array, num_still_frames=num_still_frames)
        still_frames = [[fi * frame_step for fi in fil] for fil in still_frames]
        return still_frames

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
        frame_step = self.params.get("frame_step", 1)
        num_still_frames = self.params.get("num_still_frames", 3)
        return self._detect_keyframes(
            frame_list=frame_list,
            numb_frames=len(frame_list),
            diff_threshold=diff_threshold,
            max_num_changes=max_num_changes,
            frame_step=frame_step,
            num_still_frames=num_still_frames,
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
        frame_step = self.params.get("frame_step", 1)
        num_still_frames = self.params.get("num_still_frames", 3)
        with video_capture_context(video_path=frame_path) as cap:
            numb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return self._detect_keyframes(
                frame_list=cap,
                numb_frames=numb_frames,
                diff_threshold=diff_threshold,
                max_num_changes=max_num_changes,
                frame_step=frame_step,
                num_still_frames=num_still_frames,
                disable_tqdm=disable_tqdm,
            )

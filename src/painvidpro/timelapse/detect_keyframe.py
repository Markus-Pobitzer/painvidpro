"""Code to detect keyframes in the video."""

from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from painvidpro.timelapse.utils import color_shift_recover, diff


def subsequence_colorshift(
    fixed_frame: np.ndarray, input_frame: np.ndarray, threshold: float = 0.0, percent: float = 0.5
) -> np.ndarray:
    """
    Applies color shift recovery to a subsequence of frames.

    Args:
        fixed_frame: The reference frame as a numpy array.
        input_frame: The new frame as a numpy array.
        threshold: threshold for cv2.threshold.
        percent: Percentage of non zero elements needed.

    Returns:
        np.ndarray: The output frame after color shift recovery.
    """
    # Compute the difference between the fixed frame and the input frame
    current_diff = diff(fixed_frame, input_frame)

    height, width = input_frame.shape[:2]

    # Binary mask for differences exceeding threshold
    temp = cv2.threshold(current_diff, threshold, 1.0, cv2.THRESH_BINARY)[1]

    # Check if the number of non-zero elements is greater than the specified percentage
    if cv2.countNonZero(temp) > percent * height * width:
        output_frame = color_shift_recover(input_frame, fixed_frame)
    else:
        output_frame = input_frame.copy()

    return output_frame


def get_next_number(last_num: int):
    """Generates a zero-padded string representation of a number.

    Args:
        last_num: The number to convert.

    Returns:
        str: Zero-padded string representation of the number.
    """
    return f"{last_num:04d}"


def extract_difference_mask_between_keyframe(
    keyframe_sequence_name: str,
    first_keyframe_name: str,
    keyframe_diff_threshold: float,
    keyframe_num: int,
    keyframe_mask_output_path: str,
):
    """
    Extracts the difference mask between keyframes in a video sequence.

    Args:
        keyframe_sequence_name: The name of the keyframe sequence.
        first_keyframe_name: The name of the first keyframe.
        keyframe_diff_threshold: The threshold for keyframe difference.
        keyframe_num: The number of keyframes.
        keyframe_mask_output_path: The output path for the keyframe masks.
    """
    capture = cv2.VideoCapture(keyframe_sequence_name)
    first = cv2.imread(first_keyframe_name)
    first_gaussian = cv2.GaussianBlur(first, (3, 3), 1.0)
    first_lab = cv2.cvtColor(first_gaussian, cv2.COLOR_BGR2Lab)

    for i in range(keyframe_num):
        ret, frame = capture.read()
        if not ret:
            break
        frame_gaussian = cv2.GaussianBlur(frame, (3, 3), 1.0)
        frame_lab = cv2.cvtColor(frame_gaussian, cv2.COLOR_BGR2Lab)

        diff = np.zeros(first.shape[:2], dtype=np.uint8)
        d = np.sqrt(np.sum((first_lab - frame_lab) ** 2, axis=-1))
        diff[d <= keyframe_diff_threshold] = 255

        first_lab = frame_lab.copy()
        cv2.imwrite(f"{keyframe_mask_output_path}{get_next_number(i)}.png", diff)


def prepare_intermediate_data(
    input_sequence: str, intermediate_data_path_prefix: str, do_subsequence_colorshift: bool = True
):
    """
    Prepares intermediate data for the given input sequence.

    Args:
        input_sequence: The input sequence pattern.
        intermediate_data_path_prefix: The prefix for the intermediate data path.
        do_subsequence_colorshift: If set applies subsequence_colorshift.
    """
    threshold = 10
    max_num = 50

    capture = cv2.VideoCapture(input_sequence)
    N = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    sign_array = [0] * N

    ret, frame = capture.read()
    if not ret:
        return

    img_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)

    for i in tqdm(range(1, N)):
        ret, frame = capture.read()
        if not ret:
            break
        frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        temp = cv2.absdiff(frame_lab, img_lab)
        diff = np.sqrt(np.sum(temp**2, axis=-1))

        result = np.zeros(frame.shape[:2], dtype=np.uint8)
        result[diff > threshold] = 255

        img_lab = frame_lab.copy()
        save_path = f"{intermediate_data_path_prefix}Lab_diff_binary_{get_next_number(i)}.png"
        cv2.imwrite(save_path, result)

    input1 = f"{intermediate_data_path_prefix}Lab_diff_binary_%04d.png"
    print(input1)
    capture1 = cv2.VideoCapture(input1)

    for i in range(1, N):
        ret, _frame = capture1.read()
        if not ret:
            break
        # cv2.countNonZero expects greyscale
        if _frame.ndim == 3:
            _frame = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)

        if cv2.countNonZero(_frame) < max_num:
            sign_array[i] = 0
        else:
            sign_array[i] = 1

    selected_sign = [0] * N
    selected_sign[0] = 2
    selected_sign[N - 1] = 1

    for i in range(2, N - 2):
        if sign_array[i - 1 : i + 3] == [0, 0, 0, 1]:
            selected_sign[i] = 1
        if sign_array[i - 1 : i + 3] == [1, 0, 0, 0]:
            selected_sign[i] = 2

    # Instead of creating a new VideoCapture object we can reset the reader head
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    flag = 0
    Mean = None
    averaged_index: List[int] = []

    last_positon = N
    for i in range(N - 1, 0, -1):
        if selected_sign[i] == 1:
            last_positon = i
            break

    i = 0
    while i < N:
        if selected_sign[i] == 2:
            ret, frame2 = capture.read()
            if not ret:
                break
            good_frame_name = f"{intermediate_data_path_prefix}good_original_frame_{get_next_number(i)}.png"
            cv2.imwrite(good_frame_name, frame2)
            Mean = frame2.astype(np.float32)
            j = i
            flag = 2

        if flag == 2:
            i += 1
            ret, frame2 = capture.read()
            if not ret:
                break
            good_frame_name = f"{intermediate_data_path_prefix}good_original_frame_{get_next_number(i)}.png"
            cv2.imwrite(good_frame_name, frame2)
            Mean += frame2.astype(np.float32)

            if selected_sign[i] == 1:
                averaged_index.append(1)
                Mean /= i - j + 1
                Mean = Mean.astype(np.uint8)
                save_name1 = f"{intermediate_data_path_prefix}averaged_{get_next_number(count)}.png"
                cv2.imwrite(save_name1, Mean)
                count += 1
                flag = 1

        if flag == 1:
            if i == last_positon:
                break
            i += 1
            if selected_sign[i] == 0:
                averaged_index.append(0)
                ret, frame2 = capture.read()
                if not ret:
                    break
                save_name2 = f"{intermediate_data_path_prefix}averaged_{get_next_number(count)}.png"
                cv2.imwrite(save_name2, frame2)
                count += 1

    if do_subsequence_colorshift:
        input3 = f"{intermediate_data_path_prefix}averaged_%04d.png"
        capture3 = cv2.VideoCapture(input3)
        ret, frame3 = capture3.read()
        if not ret:
            return
        output = frame3.copy()
        cv2.imwrite(f"{intermediate_data_path_prefix}subsequence_colorshift_{get_next_number(0)}.png", output)

        i = 0
        with tqdm(total=len(averaged_index), desc="Subsequence Color Shift") as pbar:
            while i < len(averaged_index):
                if averaged_index[i] == 1:
                    base = output.copy()

                ret, frame3 = capture3.read()
                if not ret:
                    break
                i += 1
                pbar.update(1)
                if i <= len(averaged_index) - 1:
                    output = subsequence_colorshift(base, frame3)
                    save_name3 = f"{intermediate_data_path_prefix}subsequence_colorshift_{get_next_number(i)}.png"
                    cv2.imwrite(save_name3, output)
                    if i == len(averaged_index) - 1:
                        break

        input4 = f"{intermediate_data_path_prefix}subsequence_colorshift_%04d.png"
    else:
        input4 = f"{intermediate_data_path_prefix}averaged_%04d.png"
    capture4 = cv2.VideoCapture(input4)
    index = 0

    # Only get the selected keyframes
    for i in range(len(averaged_index)):
        ret, frame4 = capture4.read()
        if not ret:
            break
        if i == 0:
            continue
        if averaged_index[i] == 1:
            save_name4 = f"{intermediate_data_path_prefix}subsequence_last_{get_next_number(index)}.png"
            cv2.imwrite(save_name4, frame4)
            index += 1

    print("get keyframes!")


def main():
    """
    Main function to prepare intermediate data and extract difference masks between keyframes.
    """
    root_path = "out_folder"
    input_sequence = "rose_source.mp4"
    intermediate_data_path_prefix = f"{root_path}/intermediate_data_"

    # do_subsequence_colorshift = True takes a bit of time.
    prepare_intermediate_data(input_sequence, intermediate_data_path_prefix, do_subsequence_colorshift=False)

    keyframe_sequence_name = f"{intermediate_data_path_prefix}subsequence_last_%04d.png"
    first_keyframe_name = f"{intermediate_data_path_prefix}subsequence_colorshift_0000.png"
    keyframe_diff_threshold = 8
    keyframe_num = 36
    keyframe_mask_output_path = f"{root_path}/keyframe_mask_"

    extract_difference_mask_between_keyframe(
        keyframe_sequence_name, first_keyframe_name, keyframe_diff_threshold, keyframe_num, keyframe_mask_output_path
    )

    print("get keyframe difference masks!")


if __name__ == "__main__":
    main()

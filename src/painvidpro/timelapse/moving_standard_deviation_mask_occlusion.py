"""Mioving std mask occlusion from https://github.com/CraGL/timelapse/."""

from typing import Any, List

import bottleneck
import cv2
import numpy as np
from tqdm import tqdm


def bottleneck_centered(func: Any, data: np.ndarray, window: int, axis: int = -1) -> np.ndarray:
    """Applies a bottleneck function on the data.

    Args:
        func: The bottleneck.move function.
        data: The input data.
        window: The window size, must be odd.
        axis: The axis to apply func on data.

    Returns:
        np.ndarray: The func applied to data.

    Raises:
        ValueError if window size is odd.
    """
    if window % 2 == 0:
        raise ValueError("Even window sizes not supported!")

    result = func(data, window=window, axis=axis)
    shift = window // 2
    result = np.roll(result, -shift, axis=axis)

    ## I don't know how to write a general axis selection, so let's use swapaxis
    if -1 != axis:
        result = result.swapaxes(axis, -1)
    result[..., :shift] = result[..., shift][..., np.newaxis]
    result[..., -shift - 1 :] = result[..., -shift - 1][..., np.newaxis]
    if -1 != axis:
        result = result.swapaxes(axis, -1)
    return result


def lerp_image(first: np.ndarray, last: np.ndarray, index: int, length: int):
    """Lerps two images.

    Args:
        first: The first image.
        last: The last image.
        index: The interpolation image.
        length: Number of interpolation steps between first and last image.

    Returns:
        np.ndarray: The lerped image.
    """
    interpolation = np.zeros(first.shape, dtype=np.uint8)
    if index > length:
        index = length
    interpolation = np.uint8(first * (1.0 - index * 1.0 / length) + last * (index * 1.0 / length)).clip(0, 255)
    return interpolation


def precompute_indices_for_median(window_size: int, width: int, kBufferSize: int, elem_to_right: int) -> np.ndarray:
    """Precomputes indices for all columns.

    Args:
        window_size: The window size.
        width: The image width.
        kBufferSize: The buffer size.
        elem_to_right: Number of elements taken from the right.

    Returns:
        np.ndarray: The indices.
    """
    indices = np.zeros((width, window_size), dtype=int)
    for j in range(width):
        # Left elements (up to buffersize) including j
        left_start = max(0, j - kBufferSize)
        left = np.arange(left_start, j + 1)

        # Padding left
        if kBufferSize > j:
            left = np.concat((np.zeros(kBufferSize - j), left))

        # Right elements (up to buffersize//2 + 1)
        right_start = j + 1
        right_end = min(width, j + elem_to_right + 1)
        right = np.arange(right_start, right_end)

        # Padding right
        if width <= (j + elem_to_right):
            right = np.concat((np.full((j + elem_to_right) - width + 1, width - 1), right))

        # Combine
        combined = np.concatenate([left, right])

        indices[j] = combined
    return indices


def moving_median(
    maskname: str, imgname: str, buffersize: int, outputname: str, mask_fill: int = 3, disable_tqdm: bool = True
):
    """Applies the moving median to all pixels.

    The mask and image have following format: (pixel_idx, frame_idx, rgb).
    Where:
        pixel_idx is the index of the pixel in the image (flattened).
        frame_idx is the index of the frame (relative speaking).
        rgb are the RGB values of the pixel with index pixel_idx at
        frame frame_idx.


    Args:
        maskname: Path to the mask on disk.
        imgname: Path to the image on disk.
        buffersize: Defines the window for the calculation.
        outputname: The path of the output image on disk where
            the result gets saved to.
        mask_fill: Defines how the masked pixels get filled.
            0: The previous save pixel is taken.
            1: The pixel of the first frame idx is taken.
            2: Interpolation between pixel of first frame idx and last.
        disable_tqdm: If set disables the progress bar.
    """
    mask = cv2.imread(maskname, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(imgname, 1)

    kBufferSize = int(buffersize)
    output = img.copy()
    height, width, channels = img.shape
    frame_batched = True

    if mask_fill == 0:
        for j in range(1, width - 2):
            # Take previous save pixel if current one is masked
            img[:, j, :][mask[:, j] == 255] = img[:, (j - 1), :][mask[:, j] == 255]
    elif mask_fill == 1:
        for j in range(1, width - 2):
            # Take the last safe keyframe
            img[:, j, :][mask[:, j] == 255] = img[:, 0, :][mask[:, j] == 255]
    else:
        for j in range(1, width - 2):
            # Interpolate between keyframes
            img[:, j, :][mask[:, j] == 255] = lerp_image(
                img[:, 0, :][mask[:, j] == 255], img[:, width - 1, :][mask[:, j] == 255], j, width
            )

    elem_to_right = kBufferSize // 2
    window_size = kBufferSize + elem_to_right + 1

    # For small window sizes this is fine, bigger window sizes
    # may profit from a moving/rolling median implementation instead of computing the median
    # for each window seperately.
    indices = precompute_indices_for_median(
        window_size=window_size, width=width, kBufferSize=kBufferSize, elem_to_right=elem_to_right
    )

    if frame_batched:
        # Process each row and channel
        for i in tqdm(range(height), desc="Processing rows", disable=disable_tqdm):
            for k in range(channels):
                # Gather elements using precomputed indices
                window_values = img[i, :, k][indices]
                output[i, :, k] = np.median(window_values, axis=1)
    else:
        # Seems to be slower than iterating over the frames.
        for k in tqdm(range(channels), desc="Calculating median"):
            window_values = img[:, :, k][:, indices]
            output[:, :, k] = np.median(window_values, axis=2)

    cv2.imwrite(outputname, output)


def msdmo(
    cs_video_path: str,
    keyframe_masks_path: str,
    keyframes_path: str,
    keyframe_index_list: List[int],
    output_path_prefix: str = "moving_stdev/",
    recover_image_output_path: str = "recovered/frame_",
    kbuffersize: int = 9,
    std_threshold: float = 1.5,
    std_window: int = 7,
    save_highlighted_bad_pixels: bool = False,
    disable_tqdm: bool = False,
):
    """Creates frame masks with the moving standard deviation and removes occlusions.

    The function works on frame sequences between two keyframes. It takes the std deviation and
    median of pixels over time by reshaping the sequence of [frame_idx, width, height, rgb] into
    a sequence of pixels over time/frames [width * heigth, frame_idx, rgb].

    Args:
        cs_video_path: Input frames as a cv2.VideoCapture filename, either normal or color shifted.
        keyframe_masks_path: Keyframe difference masks as a cv2.VideoCapture filename.
        keyframes_path: Keyframe images as a cv2.VideoCapture filename.
        keyframe_index_list: List of keyframe indexes related to cs_video_path.
        output_path_prefix: Output path prefix.
        recover_image_output_path: Path to store the recovered images without occlusions.
        kbuffersize: Buffersize used for median window.
        std_threshold: Threshold for standard deviation comparisons.
        std_window: Window size for compution of standard deviation.
        save_highlighted_bad_pixels: If set saves image where bad pixels are highlighted.
        disable_tqdm: If set disables the progress bar.
    """
    # capture = cv2.VideoCapture(INPUT_PATH_PREFIX + "subsequence_colorshift_%04d.png")
    capture = cv2.VideoCapture(cs_video_path)  # All frames, normally color shifted between keyframes
    capture1 = cv2.VideoCapture(keyframe_masks_path)
    capture2 = cv2.VideoCapture(keyframes_path)

    KEYFRAME_INDICES = np.array(keyframe_index_list)

    ret, firstframe = capture.read()
    # Throw away first frame, only needed if it is not a real keyframe
    # TODO: This has to be further investigated
    firstframe = np.zeros(firstframe.shape, dtype=firstframe.dtype) + 255

    first_keyframe = firstframe
    ret2, last_keyframe = capture2.read()

    count = 0
    kernel2 = np.ones((3, 3), np.uint8)

    # Not implemented
    use_biliteral_filter = False

    running_index = 0
    for index in tqdm(range(1, KEYFRAME_INDICES.shape[0]), disable=disable_tqdm):
        if index == 1:
            # Mask everything as changed, only needed if we do not have first real keyframe
            mask = np.zeros(firstframe.shape, dtype=np.uint8) + 255
        else:
            retval1, mask = capture1.read()
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

        cv2.imwrite(output_path_prefix + "keyframe_mask_closed_" + "{0:04}".format(index) + ".png", mask)

        nonzero_num = cv2.countNonZero(mask)
        kf_indx_dist = KEYFRAME_INDICES[index] - KEYFRAME_INDICES[index - 1]

        if nonzero_num < mask.shape[0] * mask.shape[1]:
            zero_num = firstframe.shape[0] * firstframe.shape[1] - nonzero_num
            # Image has as first shape the number of pixels that are zero in the mask (i.e. no change?)
            # As second dimension it has the number of frames between the two keyframes and a buffer
            # The RGB pixels as last dimension
            img = np.zeros((zero_num, kf_indx_dist + kbuffersize + 1, 3), dtype=np.uint8)
            temp = np.zeros((firstframe.shape[0], firstframe.shape[1], 3), dtype=np.uint8)

            # For the frames between the two keyframes
            for i in range(0, kf_indx_dist + 1):
                if i == 0:
                    frame = firstframe
                if i >= 1:
                    retval, frame = capture.read()
                    running_index += 1
                frame_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                # Interesting that we do not need to reshape it to (zero_num, 1, 3)
                img[:, i, :] = (frame_lab[mask == 0]).reshape((zero_num, 3))
                # So temp is just the last frame since we always overwrite it
                # This is equal to the next keyframe (+ 1 in the loop)
                temp = frame_lab

            # For the buffer frames
            for i in range(
                kf_indx_dist + 1,
                kf_indx_dist + kbuffersize + 1,
            ):
                # Overwrite the buffer inices with the last frame
                img[:, i, :] = (temp[mask == 0]).reshape((zero_num, 3))

            firstframe = cv2.cvtColor(temp, cv2.COLOR_LAB2BGR)

            img_RGB = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

            subsequence_img_name = output_path_prefix + "subsequence_img_" + "{0:04}".format(index) + ".png"
            cv2.imwrite(subsequence_img_name, img_RGB)

            # Okay bottleneck.move_std gets applied to img somehow
            stdevs = bottleneck_centered(bottleneck.move_std, img, window=std_window, axis=1)

            stdevs_per_pixel = stdevs.sum(axis=2) / (stdevs.shape[2])

            badmask = np.zeros((zero_num, kf_indx_dist + kbuffersize + 1), dtype=np.uint8)

            # Mask of pixels where the stdev id high
            badmask[stdevs_per_pixel > std_threshold] = 255

            subsequence_mask_name = output_path_prefix + "subsequence_mask_" + "{0:04}".format(index) + ".png"
            cv2.imwrite(subsequence_mask_name, badmask)

            if save_highlighted_bad_pixels:
                highlighted_bad_pixels = img_RGB.copy()
                highlighted_bad_pixels[stdevs_per_pixel > std_threshold] = (255, 0, 0)
                subsequence_highlighted_name = (
                    output_path_prefix + "subsequence_highlighted_" + "{0:04}".format(index) + ".png"
                )
                cv2.imwrite(subsequence_highlighted_name, highlighted_bad_pixels)

            subsequence_mm_outputname = "mm_with_mask/" + "frame_" + "{0:04}".format(index) + ".png"

            # Update the pixels with their moving median
            moving_median(subsequence_mask_name, subsequence_img_name, kbuffersize, subsequence_mm_outputname)

            frame_mask = mask
            recover_base = cv2.imread(subsequence_mm_outputname)

            if use_biliteral_filter:
                # TODO: CV2 removed the bilateral filter, update to other function.
                # using bilateral filtering
                temp = np.zeros(
                    (recover_base.shape[0] * 3, recover_base.shape[1], recover_base.shape[2]), dtype=np.uint8
                )
                for i in range(0, recover_base.shape[0]):
                    temp[3 * i + 0, :, :] = recover_base[i, :, :]
                    temp[3 * i + 1, :, :] = recover_base[i, :, :]
                    temp[3 * i + 2, :, :] = recover_base[i, :, :]

                temp2 = cv2.adaptiveBilateralFilter(temp, (3, 15), 200.0)

                for i in range(0, recover_base.shape[0]):
                    recover_base[i, :, :] = temp2[3 * i + 1, :, :]

            _frame = np.zeros(firstframe.shape, dtype=np.uint8)
            for i in range(0, kf_indx_dist + kbuffersize + 1):
                _frame = lerp_image(first_keyframe, last_keyframe, i, kf_indx_dist)
                _frame[frame_mask == 0] = (recover_base[:, i, :]).reshape((zero_num, 3))
                cv2.imwrite(recover_image_output_path + "{0:04}".format(count) + ".png", _frame)
                count = count + 1

        else:
            # This means that no pixels changed between the previous and current keyframe
            _frame = np.zeros(firstframe.shape, dtype=np.uint8)
            frame_mask = mask
            for i in range(0, kf_indx_dist + 1):
                if i == 0:
                    frame = firstframe
                if i >= 1:
                    retval, frame = capture.read()
                    running_index += 1
                firstframe = frame

            for i in range(0, kf_indx_dist + 1):
                _frame = lerp_image(first_keyframe, last_keyframe, i, kf_indx_dist)
                cv2.imwrite(recover_image_output_path + "{0:04}".format(count) + ".png", _frame)
                count = count + 1

        first_keyframe = last_keyframe
        ret2, last_keyframe = capture2.read()

        if not ret2:
            print("No keyframe at index", index)
            break

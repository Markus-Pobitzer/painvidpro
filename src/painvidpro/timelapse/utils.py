"""Utility functions."""

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


# Original code: https://github.com/CraGL/timelapse/


def to_lab(bgr: np.ndarray):
    """
    Converts a BGR image to a normalized float32 format.

    Args:
        bgr: The input BGR image.

    Returns:
        np.ndarray: The normalized image.
    """
    bgr32 = bgr.astype(np.float32) / 255.0
    return bgr32


def diff(mat1: np.ndarray, mat2: np.ndarray):
    """
    Computes the difference between two images in color space.

    Args:
        mat1: The first image.
        mat2: The second image.

    Returns:
        np.ndarray: The difference image.
    """
    mat1_lab = to_lab(mat1)
    mat2_lab = to_lab(mat2)
    diff = cv2.absdiff(mat1_lab, mat2_lab)

    # Optimization of https://github.com/CraGL/timelapse\
    # /blob/master/1%20preprocessing/s1_whole_sequence_colorshift\
    # /colorshift_filter.cpp#L212
    diff = np.sqrt(np.sum(diff**2, axis=2)) / 1.8
    return (diff * 255).astype(np.uint8)


def get_mask(before: np.ndarray, after: np.ndarray):
    """Generates a mask based on color differences.

    Those pixels get masked where the difference between the frames
    are between the 8th and 4th percentile.

    Args:
        before: The reference frame.
        after: The current frame.

    Returns:
        The mask as a np.ndarray.
    """
    height, width = before.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    before = before.astype(np.float64) / 255.0
    after = after.astype(np.float64) / 255.0
    diff = np.sum((before - after) ** 2, axis=2)

    diff_list = [(diff[i, j], i, j) for i in range(diff.shape[0]) for j in range(diff.shape[1])]
    diff_list.sort()

    # To get the indices between the 8th and 4th percentile
    N = len(diff_list)
    for _, i, j in diff_list[N // 8 : N // 4]:
        mask[i, j] = 1
    return mask


def solve_lsm(old_img: np.ndarray, new_img: np.ndarray, mask: np.ndarray):
    """
    Solves the least squares problem for color correction.

    Args:
        old_img: The reference image.
        new_img: The current image.
        mask: The mask for selected pixels.

    Returns:
        np.ndarray: The corrected image.
    """
    old_img = old_img.astype(np.float64) / 255.0
    new_img = new_img.astype(np.float64) / 255.0

    mask_indices = np.where(mask == 1)
    count = len(mask_indices[0])

    new_sum = np.sum(new_img[mask_indices])
    old_sum = np.sum(old_img[mask_indices])
    new_sum_square = np.sum(new_img[mask_indices] ** 2)
    old_new_sum = np.sum(new_img[mask_indices] * old_img[mask_indices])

    M = csr_matrix([[count, new_sum], [new_sum, new_sum_square]])
    N = np.array([old_sum, old_new_sum])

    # Solve MX = N
    X = spsolve(M, N)

    recovered = old_img.copy()
    img_diff_mask = np.where(abs(old_img - new_img) >= 0)
    recovered[img_diff_mask] = X[0] + X[1] * new_img[img_diff_mask]
    recovered = np.clip(recovered, 0, 1)

    return (recovered * 255).astype(np.uint8)


def color_shift_recover(input_frame: np.ndarray, fixed_frame: np.ndarray) -> np.ndarray:
    """
    Recovers the color shift between the input frame and the fixed frame.

    Args:
        input_frame: The new frame as a numpy array.
        fixed_frame: The reference frame as a numpy array.

    Returns:
        np.ndarray: The recovered frame.
    """
    mask = get_mask(fixed_frame, input_frame)
    # Split the fixed frame into BGR channels
    channels = cv2.split(fixed_frame)
    # Split the input frame into BGR channels
    new_channels = cv2.split(input_frame)
    recovered_channels = []

    for old_channel, new_channel in zip(channels, new_channels):
        recovered_channel = solve_lsm(old_channel, new_channel, mask)
        recovered_channels.append(recovered_channel)

    recovered_frame = cv2.merge(recovered_channels)

    return recovered_frame


def extract_frames(input_video_path: str, output_video_path: str, start_frame: int, end_frame: int):
    """
    Extracts frames from a video and saves them as a new video.

    Args:
        input_video_path: The path to the input video file.
        output_video_path: The path to the output video file.
        start_frame: The starting frame number.
        end_frame: The ending frame number.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    # Get the frame rate of the input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the width and height of the frames in the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Set the starting frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read and write frames from start_frame to end_frame
    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

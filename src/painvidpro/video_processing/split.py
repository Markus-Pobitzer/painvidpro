"""Utiltiy function to split a video."""

import cv2
import numpy as np

from painvidpro.video_processing.utils import video_capture_context


def estimate_split_width(image: np.ndarray, left_border: float = 0.0, right_border: float = 1.0) -> int:
    """Returns the column/width value where the image can be split vertically.

    The assumption is that the image is a concatination of two images
    and this function tries to find the pixel value (from the left), where
    the two iamges are concatenated.

    Args:
        image: The image as an numpy array.
        left_border: value in range [0, 1]. All vlaues left of left_border
            are set to 0 and have no influence.
        right_border: value in range [0, 1]. All values right of right_border
            are set to 0 and have no influence.

    Returns:
        The pixel value from the left.
    """
    kernel = np.array([[1, 0, -1]])
    # Applying the kernel, allowing the results to be negative.
    dst = cv2.filter2D(image, cv2.CV_64F, kernel)
    # We just care about the change not the direction of the gradient.
    dst = np.absolute(dst)
    dst = dst.max(axis=-1)

    # Find the column with the biggest value
    col_sum = np.sum(dst, axis=0)
    # Set values outside of the borders to 0
    w = col_sum.size
    col_sum[: int(left_border * w)] = 0
    col_sum[int(right_border * w) :] = 0
    # Return index of biggest value
    argmax = np.argmax(col_sum)
    return argmax


def estimate_split_width_from_frame(
    video_file_path: str, frame_index: int = 0, left_border: float = 0.0, right_border: float = 1.0
) -> int:
    """Estimates the split width based on a specific frame in video_file_path.

    If frame index is not specified or set to a value smaller than 0, the
    first frame of the video is taken, equivalent to frame_index = 0.

    Args:
        video_file_path: String of the video path.
        frame_index: The frame to choose from video_file_path.
        left_border: value in range [0, 1]. All vlaues left of left_border
            are set to 0 and have no influence.
        right_border: value in range [0, 1]. All values right of right_border
            are set to 0 and have no influence.

    Returns:
        Estimate of the split width of the video.
    """
    if frame_index < 0:
        frame_index = 0

    idx = 0
    with video_capture_context(video_path=video_file_path) as cap:
        while True:
            res, frame = cap.read()
            if not res:
                break
            if idx == frame_index:
                split_width = estimate_split_width(frame, left_border=left_border, right_border=right_border)
                # Adding one pixel, it leads to better results
                split_width += 1
                return split_width
            idx += 1
    return -1


def split_horizontal(
    video: str, output_left: str, output_right: str, split_width: int = -1, cv2_video_format: str = "mp4v"
):
    """Splits a video horizonatal based on given ratio (from the left side).

    Args:
        video: The path to the input video.
        output_left: The output video of the left side.
            Does not create an output for the left side if output_left
            is the empty string.
        output_right: The output video of the right side.
            Does not create an output for the right side if output_right
            is the empty string.
        split_width: The split width from the left.
        cv2_video_format: CV2 video format.

    Raises:
        ValueError if ratio is not in range [0, 1].
    """
    fourcc = cv2.VideoWriter_fourcc(*cv2_video_format)

    with video_capture_context(video_path=video) as cap:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if split_width < 0:
            split_width = width // 2
        elif split_width > width:
            split_width = width

        width_left = split_width
        width_right = width - width_left

        if output_left != "":
            writer_left = cv2.VideoWriter(output_left, fourcc, fps, (width_left, height))

        if output_right != "":
            writer_right = cv2.VideoWriter(output_right, fourcc, fps, (width_right, height))

        try:
            ret, frame = cap.read()
            while ret:
                if output_left != "":
                    frame_left = frame[:, :width_left]
                    writer_left.write(frame_left)

                if output_right != "":
                    frame_right = frame[:, width_left:]
                    writer_right.write(frame_right)

                ret, frame = cap.read()

        finally:
            if output_left != "":
                writer_left.release()

            if output_right != "":
                writer_right.release()

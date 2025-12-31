"""Class for the Loomis Keyframe detection."""

import os
from pathlib import Path
from typing import Any, Dict

import cv2

from painvidpro.data_storage.hdf5_video_archive import DynamicVideoArchive
from painvidpro.processors.sam3_matting import ProcessorSAM3
from painvidpro.video_processing.split import estimate_split_width_from_frame
from painvidpro.video_processing.utils import video_capture_context, video_writer_context


class ProcessorSAM3Loomis(ProcessorSAM3):
    def __init__(self):
        """Class to process videos."""
        super().__init__()

    def detect_canvas(
        self, video_dir: Path, video_path: str, frame_data: DynamicVideoArchive, canvas_erosion: int = 2
    ) -> bool:
        """Detects a canvas region in the video, crops the video to this region, and updates metadata.

        If the metadata indicates the canvas has already been detected, the function exits early. Otherwise,
        it processes the video starting from the frame specified in the metadata to detect the canvas.
        The canvas gets detected by estimating at which vertical line the pixel change is greatest and taking the right side.
        Overwrites the original video file. Metadata is updated to reflect the detection status.

        Args:
            video_dir: Path to the directory containing the video.
            video_path: Path to the video file.
            frame_data: DynamicVideoArchive containting the metadata dictionary:
                      - "start_frame_idx": Frame index to start canvas detection.
                      - "canvas_detected": Flag indicating if the canvas was already detected.
                      The dictionary is modified add/update the "canvas_detected" key.
            canvas_erosion: Indicates by how many pixels the canvas should be eroded.

        Returns:
            bool: Always returns True, indicating the process completed (successful detection or fallback to original video).
                  Returns immediately if the canvas was already pre-detected.

        Side Effects:
            - Overwrites the original video with the cropped version if a canvas is detected.
            - Modifies the metadata dictionary to set "canvas_detected" = True on successful detection.
            - Persists updated metadata to disk.
        """
        with frame_data:
            metadata: Dict[str, Any] = frame_data.get_global_metadata()  # type: ignore
            # In case we already detected the canvas
            if metadata.get("canvas_detected", False):
                return True
            start_frame_idx = int(metadata["start_frame_idx"])

        try:
            # Estimating the split width
            split_width = estimate_split_width_from_frame(
                video_file_path=video_path, frame_index=start_frame_idx, left_border=0.4, right_border=0.6
            )
        except Exception as e:
            self.logger.info(
                (
                    f" Failed to estimate split_width from video {video_path} "
                    f"with frame_index = {start_frame_idx}:\nException: {e}."
                )
            )
            return False

        cavas_only_video_path = str(video_dir / "canvas_only_video.mp4")
        with video_capture_context(video_path=video_path) as cap:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            # Convert to integers (pixel indices)
            if split_width < 0:
                split_width = width // 2
            x_min = max(0, int(round(split_width) + canvas_erosion))
            y_min = 0
            x_max = width
            y_max = height
            new_width = x_max - x_min
            new_height = y_max - y_min

            with video_writer_context(
                output_path=cavas_only_video_path, width=new_width, height=new_height, fps=fps
            ) as vid_out:
                ret, frame = cap.read()
                while ret:
                    # Crop the image
                    cropped_image = frame[y_min:y_max, x_min:x_max]
                    vid_out.write(cropped_image)
                    ret, frame = cap.read()

        # Overwrite the original video with the cropped one
        os.replace(src=cavas_only_video_path, dst=video_path)
        with frame_data:
            frame_data.set_global_metadata("canvas_detected", True)
            frame_data.set_global_metadata("canvas_coordinates", [x_min, y_min, x_max, y_max])
        self.logger.info(
            (
                f"Detected Canvas with coordinates ({x_min}, {y_min}), ({x_max}, {y_max}).\n"
                "The original video was overwritten with the cropped version."
            )
        )
        return True

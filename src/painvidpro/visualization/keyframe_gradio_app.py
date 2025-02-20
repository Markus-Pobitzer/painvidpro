"""Visualization for detected keyframes."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np

from painvidpro.utils.metadata import load_metadata
from painvidpro.visualization.utils import get_keyframe_metadata, get_video_folders


def load_pipeline(root_folder: str):
    """Loads the pipeline .json."""
    with open(os.path.join(root_folder, "pipeline.json"), "r") as f:
        pipeline = json.load(f)
    return pipeline


def filter_processed_metadata(sub_subfolders: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """Function to filter metadata based on entries."""
    processed_metadata = []
    for sub_subfolder in sub_subfolders:
        succ, metadata = load_metadata(Path(sub_subfolder))
        if not succ:
            continue
        (start_frame, end_frame), keyframe_list, selected_keyframe_list = get_keyframe_metadata(metadata)
        # Only take samples that have been successfully processed with the Keyframe Processor
        if start_frame < 0 or end_frame < 0 or keyframe_list is None or len(selected_keyframe_list) < 2:
            continue
        processed_metadata.append((sub_subfolder, metadata))
    return processed_metadata


def get_video_path(sub_subfolder, video_name: str = "video.mp4"):
    """Returns the video path."""
    return os.path.join(sub_subfolder, video_name)


def load_video_and_keyframes(
    sub_subfolder: str, metadata: Dict[str, Any], video_name: str = "video.mp4"
) -> Tuple[str, List[int]]:
    """Function to load video and keyframes"""
    video_path = os.path.join(sub_subfolder, video_name)
    keyframes = metadata.get("selected_keyframe_list", [])
    return video_path, keyframes


def display_keyframes(video_path: str, keyframes: List[int]) -> List[np.ndarray]:
    """Function to display keyframes in RGB space."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    for frame_idx in keyframes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    cap.release()
    return frames


def get_frame(video_path: str, frame_idx: int) -> np.ndarray:
    """Loads the frame from the video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Was not able to laod frame with index {frame_idx} from {video_path}")
    # Convert BGR to RGB
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def compute_progress_dist(metadata: Dict[str, Any], number_bins: int = 100) -> List[int]:
    """Each Video from start frame to last keaframe gets ordered in to bins."""
    start_frame_idx = metadata["start_frame_idx"]
    last_keyframe_idx = metadata["selected_keyframe_list"][-1] - start_frame_idx
    progress_bin_list = [0] * number_bins
    for sele_keyframe in metadata["selected_keyframe_list"][:-1]:
        progress = (sele_keyframe - start_frame_idx) / last_keyframe_idx
        prog_bin = int(progress * number_bins)
        progress_bin_list[prog_bin] = 1
    return progress_bin_list


def vis_progress_distribution(progress_dist: List[int], width=1000, height=25) -> np.ndarray:
    """Creates a visual representation of the progress distribution."""
    n = len(progress_dist)
    bin_width = (width - (n - 1) * 2) // n  # Calculate the width of each bin
    image = np.zeros((height, width, 3), dtype=np.uint8)

    current_x = 0
    for value in progress_dist:
        color = (0, 255, 0) if value == 1 else (255, 127, 127)  # Green for 1, Red for 0
        image[:, current_x : current_x + bin_width] = color
        current_x += bin_width
        if current_x < width:
            image[:, current_x : current_x + 2] = (255, 255, 255)  # White vertical line
            current_x += 2

    return image


def get_reference_frame(reference_frame_path: str, video_path: str, keyframes: List[int]) -> np.ndarray:
    """Loads the reference frame."""
    img = cv2.imread(reference_frame_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    try:
        return get_frame(video_path, keyframes[-1])
    except Exception as _:
        return np.zeros((250, 250, 3)) + (255, 0, 0)


# Gradio app
def gradio_app(root_folder: str):
    """The Gradio App for visualization of the Keyframes."""
    pipeline = load_pipeline(root_folder)
    numb_video = 0
    for key in pipeline.get("video_item_dict", {}).keys():
        numb_video += len(pipeline["video_item_dict"][key].keys())
    sub_subfolders = get_video_folders(root_folder)
    processed_metadata = filter_processed_metadata(sub_subfolders)

    def update_display(selected_index):
        sub_subfolder, metadata = processed_metadata[selected_index]
        video_name = metadata.get("processed_video_name", "video.mp4")
        reference_frame_name = metadata.get("reference_frame_name", "reference_frame.png")
        reference_frame_path = os.path.join(sub_subfolder, reference_frame_name)
        video_path, keyframes = load_video_and_keyframes(sub_subfolder, metadata, video_name=video_name)
        if len(keyframes) > 0:
            try:
                ret_keyframe = get_frame(video_path, keyframes[0])
            except Exception as _:
                ret_keyframe = np.zeros((250, 250, 3)) + (255, 0, 0)
        else:
            ret_keyframe = np.zeros((250, 250, 3))

        # To show how well distributed the keyframes are
        progress_bin_list = compute_progress_dist(metadata)
        prog_dist_img = vis_progress_distribution(progress_bin_list)
        reference_frame = get_reference_frame(reference_frame_path, video_path, keyframes)
        # return video_path, ret_keyframe
        return reference_frame, ret_keyframe, prog_dist_img

    with gr.Blocks() as demo:
        gr.Markdown("## Video and Keyframe Viewer")

        pipe_panel = gr.Accordion("Pipeline Information", open=False)
        with pipe_panel:
            video_path_text = gr.Textbox(label="Root folder", value=root_folder)
            start_frame_idx_text = gr.Textbox(label="Number of videos", value=numb_video)

        selected_index = gr.Number(label="Select Index", value=0)

        info_panel = gr.Accordion("Video Information", open=False)
        with info_panel:
            video_path_text = gr.Textbox(label="Video Path")
            channel_text = gr.Textbox(label="Channel")
            art_media_text = gr.Textbox(label="Art Media")
            numb_frames_text = gr.Textbox(label="Number of Frames")
            start_frame_idx_text = gr.Textbox(label="Start Frame Index")
            end_frame_idx_text = gr.Textbox(label="End Frame Index")

        # video_output = gr.Video(label="Video")
        keyframe_slider = gr.Slider(label="Keyframes", minimum=0, maximum=0, step=1)
        image_progress = gr.Image(label="Progress distribution", type="numpy")
        with gr.Row():
            image_reference = gr.Image(label="Reference Frame", type="numpy")
            image_output = gr.Image(label="Selected Keyframe", type="numpy")

        prev_button = gr.Button("Previous")
        next_button = gr.Button("Next")

        def update_keyframe_slider(selected_index):
            sub_subfolder, metadata = processed_metadata[selected_index]
            video_name = metadata.get("processed_video_name", "video.mp4")
            _, keyframes = load_video_and_keyframes(sub_subfolder, metadata, video_name=video_name)
            return gr.update(maximum=len(keyframes) - 1, value=0)

        def update_image_output(selected_index, keyframe_index):
            sub_subfolder, metadata = processed_metadata[selected_index]
            video_name = metadata.get("processed_video_name", "video.mp4")
            video_path, keyframes = load_video_and_keyframes(sub_subfolder, metadata, video_name=video_name)
            try:
                ret_keyframe = get_frame(video_path, keyframes[keyframe_index])
            except Exception as _:
                ret_keyframe = np.zeros((250, 250, 3)) + (255, 0, 0)
            return ret_keyframe

        def update_info_panel(selected_index):
            sub_subfolder, metadata = processed_metadata[selected_index]
            video_name = metadata.get("processed_video_name", "video.mp4")
            video_path = get_video_path(sub_subfolder, video_name=video_name)
            channel = metadata.get("channel", "")
            art_media = "; ".join(metadata.get("art_media", []))
            numb_frames = metadata.get("number_frames", -1)
            start_frame_idx = metadata.get("start_frame_idx", -1)
            end_frame_idx = metadata.get("end_frame_idx", -1)
            return video_path, channel, art_media, numb_frames, start_frame_idx, end_frame_idx

        prev_button.click(lambda x: max(0, x - 1), inputs=selected_index, outputs=selected_index)
        next_button.click(
            lambda x: min(len(processed_metadata) - 1, x + 1), inputs=selected_index, outputs=selected_index
        )

        # selected_index.change(update_display, inputs=selected_index, outputs=[video_output, image_output])
        selected_index.change(
            update_display, inputs=selected_index, outputs=[image_reference, image_output, image_progress]
        )
        selected_index.change(update_keyframe_slider, inputs=selected_index, outputs=[keyframe_slider])
        selected_index.change(
            update_info_panel,
            inputs=selected_index,
            outputs=[
                video_path_text,
                channel_text,
                art_media_text,
                numb_frames_text,
                start_frame_idx_text,
                end_frame_idx_text,
            ],
        )

        keyframe_slider.change(update_image_output, inputs=[selected_index, keyframe_slider], outputs=image_output)

    demo.launch(server_name="0.0.0.0", server_port=7860)

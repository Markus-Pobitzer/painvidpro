"""Visualization for detected keyframes."""

import os

import gradio as gr
import numpy as np

from painvidpro.visualization.utils import (
    compute_progress_dist,
    filter_processed_metadata,
    get_frame,
    get_reference_frame,
    get_video_folders,
    get_video_path,
    load_pipeline,
    load_video_and_keyframes,
    vis_progress_distribution,
)


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
            root_folder_text = gr.Textbox(label="Root folder", value=root_folder)  # noqa: F841
            numb_videos_text = gr.Textbox(label="Number of videos", value=numb_video)  # noqa: F841

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

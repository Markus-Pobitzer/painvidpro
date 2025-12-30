"""Visualization extracted frames."""

from typing import List, Optional

import gradio as gr
import numpy as np

from painvidpro.visualization.utils import (
    cleanup,
    create_temp_file,
    filter_processed_metadata_extracted_frames,
    get_ref_frame_variations,
    get_reference_frame,
    get_video_folders,
    get_video_path,
    load_log_files,
    load_pipeline,
    read_file,
    save_video_from_frames,
    update_exclude_video_flag,
)


# Gradio app
def gradio_app(root_folder: str):
    """The Gradio App for visualization of the extracted frames."""
    pipeline = load_pipeline(root_folder)
    numb_video = 0
    for key in pipeline.get("video_item_dict", {}).keys():
        numb_video += len(pipeline["video_item_dict"][key].keys())
    sub_subfolders = get_video_folders(root_folder)
    processed_metadata = filter_processed_metadata_extracted_frames(sub_subfolders)
    tmp_video_path = create_temp_file()

    def update_display(selected_index):
        sub_subfolder, metadata = processed_metadata[selected_index]
        exclude_video = metadata.get("exclude_video", False)
        reference_frame = get_reference_frame(sub_subfolder)

        # return video_path, ret_keyframe
        return exclude_video, reference_frame

    with gr.Blocks() as demo:
        gr.Markdown("## Video and Keyframe Viewer")

        pipe_panel = gr.Accordion("Pipeline Information", open=False)
        with pipe_panel:
            gr.Textbox(label="Root folder", value=root_folder)  # noqa: F841
            gr.Textbox(label="Number of videos", value=numb_video)  # noqa: F841
            frame_skip_video_input = gr.Number(label="Frame skip for `Extracted Frame Video`", value=10)

        selected_index = gr.Number(label="Select Index", value=0)

        with gr.Row():
            prev_button = gr.Button("< Previous")
            next_button = gr.Button("Next >")

        info_panel = gr.Accordion("Video Information", open=False)
        with info_panel:
            video_path_text = gr.Textbox(label="Video Path")
            channel_text = gr.Textbox(label="Channel")
            art_media_text = gr.Textbox(label="Art Media")
            numb_frames_text = gr.Textbox(label="Number of Frames")
            start_frame_idx_text = gr.Textbox(label="Start Frame Index")
            end_frame_idx_text = gr.Textbox(label="End Frame Index")

        with gr.Row():
            exclude_video_checkbox = gr.Checkbox(label="Exclude Video")

        with gr.Row():
            image_reference = gr.Image(label="Reference Frame", type="numpy", height=512)
            video_output = gr.Video(label="Extracted Frame Video", height=512)

        various_ref_gallery = gr.Gallery(label="Image Grid", columns=3)

        log_out_text = gr.Textbox(label="Log", lines=40)

        def update_exclude_video(selected_index, checkbox_value):
            """Handles the exclude video checkobox."""
            sub_subfolder, _ = processed_metadata[selected_index]
            try:
                metadata = update_exclude_video_flag(sub_subfolder=sub_subfolder, exclude_video=checkbox_value)
                processed_metadata[selected_index] = (sub_subfolder, metadata)
            except Exception as e:
                raise gr.Error(f"Was not able to save metadata in folder {sub_subfolder}: {e}")

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

        def update_video(selected_index, frame_skip_video_input) -> Optional[str]:
            sub_subfolder, _ = processed_metadata[selected_index]
            return save_video_from_frames(
                video_dir=sub_subfolder, video_output_path=tmp_video_path, frame_skip_video=int(frame_skip_video_input)
            )

        def update_ref_frame_variations(selected_index) -> List[np.ndarray]:
            """Loads the reference frame variations if there are any."""
            sub_subfolder, _ = processed_metadata[selected_index]
            return get_ref_frame_variations(sub_subfolder=sub_subfolder)

        def get_logs(selected_index) -> str:
            """Function to dynamically generate the tabs based on the current log_files"""
            if selected_index < 0:
                return "No Log file found"
            ret = ""
            sub_subfolder, _ = processed_metadata[selected_index]
            logfile_list = load_log_files(sub_subfolder)
            for tab_name, file_path in logfile_list:
                log_content = read_file(file_path)
                ret += tab_name + ":\n" + log_content + "\n\n"

            return ret

        prev_button.click(lambda x: max(0, x - 1), inputs=selected_index, outputs=selected_index)
        next_button.click(
            lambda x: min(len(processed_metadata) - 1, x + 1), inputs=selected_index, outputs=selected_index
        )

        exclude_video_checkbox.change(update_exclude_video, inputs=[selected_index, exclude_video_checkbox])

        # selected_index.change(update_display, inputs=selected_index, outputs=[video_output, image_output])
        selected_index.change(
            update_display,
            inputs=selected_index,
            outputs=[exclude_video_checkbox, image_reference],
        )
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
        selected_index.change(update_video, inputs=[selected_index, frame_skip_video_input], outputs=[video_output])
        selected_index.change(update_ref_frame_variations, inputs=selected_index, outputs=[various_ref_gallery])
        selected_index.change(get_logs, inputs=selected_index, outputs=[log_out_text])

        demo.unload(fn=lambda: cleanup(tmp_video_path))

    demo.launch(server_name="0.0.0.0", server_port=7860)

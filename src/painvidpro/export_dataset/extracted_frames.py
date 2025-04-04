"""Exporting data created by the Matting processor."""

import copy
import json
import logging
import os.path
import random
import shutil
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Tuple

from painvidpro.pipeline.input_file_format import VideoItem
from painvidpro.pipeline.pipeline import Pipeline
from painvidpro.utils.metadata import load_metadata


logger = logging.getLogger(__name__)
random.seed(42)


def perpare_data(
    video_dir: str, source: str, video_id: str, video_item: VideoItem, metadata: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Prepares and extrcts the needed information.

    Args:
        video_dir: The path to the video directory.
        source: The video source, i.e. youtube.
        video_id: The identifier of the video.
        video_item: Pipeline information for the video.
        metadata: The metadata of the video.

    Returns:
        A bool indicating success, a Dict containing all neccessary information.
    """
    start_frame = metadata.get("start_frame_idx", -1)
    end_frame = metadata.get("end_frame_idx", -1)
    extracted_frames = metadata.get("extracted_frames", [])
    exclude_video = metadata.get("exclude_video", False)

    if exclude_video:
        logger.info((f"Excluding {video_dir}, since the exclude_video is set in the metadata."))
        return False, {}

    # Only take samples that have been successfully processed with the Keyframe Processor
    if start_frame < 0 or end_frame < 0 or len(extracted_frames) == 0:
        logger.info(
            (
                f"Failed to load {video_dir}, not enough information in metadata!\n"
                f"Start frame index must be positie: {start_frame}.\n"
                f"End frame index must be positive: {end_frame}.\n"
                f"Extracted frames must not be empty: {extracted_frames}."
            )
        )
        return False, {}

    identifier = source + "-" + video_id
    ret: Dict[str, Any] = {
        # Will be dropped before exporting to disk
        "video_dir": video_dir,
        "identifier": identifier,
        "start_frame_idx": start_frame,
        "end_frame_idx": end_frame,
        "reference_frame_name": metadata.get("reference_frame_name", "reference_frame.png"),
        "extracted_frames": extracted_frames,
        # These get saved for every frame
        "source": source,
        "video_url": video_item.url,
        "video_title": metadata.get("titel", ""),
        "video_license": metadata.get("license", ""),
        "video_channel": metadata.get("channel", ""),
        "video_channel_id": metadata.get("video_channel_id", ""),
        "art_style": video_item.art_style,
        "art_genre": video_item.art_genre,
        "art_media": video_item.art_media,
    }
    return True, ret


def save_frames_to_disk(data_list: List[Dict[str, Any]], output_path: str, split: str):
    """Saves the frames to disk and creates the dataset.

    Args:
        data_list: The list of data dicts.
        output_path: The path where to store the dataset.
        split: The split.
    """
    root_path = Path(output_path)
    root_path.mkdir(parents=True, exist_ok=True)
    split_dir = root_path / split
    split_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = root_path / f"{split}.jsonl"
    extracted_frame_list: List[Dict[str, Any]] = []

    for data_dict in data_list:
        video_dir = data_dict.pop("video_dir")
        video_path = Path(video_dir)
        identifier = data_dict.pop("identifier")
        start_frame = data_dict.pop("start_frame_idx")
        end_frame = data_dict.pop("end_frame_idx")
        reference_frame_name = data_dict.pop("reference_frame_name")
        extracted_frames = data_dict.pop("extracted_frames")

        # Reference frame
        reference_frame_path = video_path / reference_frame_name
        reference_frame_identifier = f"{identifier}-reference_frame.png"
        reference_frame_dst = split_dir / reference_frame_identifier
        if not os.path.isfile(reference_frame_path):
            logger.info(f"Was not able to load reference frame {str(reference_frame_path)}.")
            continue
        try:
            shutil.copy2(reference_frame_path, reference_frame_dst)
        except Exception as e:
            logger.info((f"Was not able to copy reference frame {str(reference_frame_path)}." f"Error was: {e}"))
            continue

        # Extracted Frames
        for extracted_frame_dict in extracted_frames:
            data_dict_frame = copy.deepcopy(data_dict)
            frame_idx = extracted_frame_dict.get("index", -1)
            frame_rel_path = extracted_frame_dict.get("path", "")
            if frame_idx < 0 or frame_rel_path == "":
                logger.info((f"Was not able to save frame {extracted_frame_dict} from" f" video dir {video_dir}."))
                continue
            frame_path = video_path / frame_rel_path
            frame_identifier = f"{identifier}-{os.path.basename(frame_rel_path)}"
            frame_path_dst = split_dir / frame_identifier
            try:
                shutil.copy2(frame_path, frame_path_dst)
            except Exception as e:
                logger.info(
                    (f"Was not able to copy frame {str(frame_path)}" f" to {str(frame_path_dst)}." f"Error was: {e}")
                )
                continue
            progress = frame_idx / (end_frame - start_frame)
            data_dict_frame["frame_progress"] = progress
            data_dict_frame["frame_path"] = str(Path(split) / frame_identifier)
            data_dict_frame["reference_frame_name"] = str(Path(split) / reference_frame_identifier)
            extracted_frame_list.append(data_dict_frame)

    # Save dataset jsonl
    with open(jsonl_file, "w", encoding="utf-8") as file:
        for item in extracted_frame_list:
            json_line = json.dumps(item)
            file.write(json_line + "\n")

    logger.info(f"{split} Split contains {len(extracted_frame_list)} frames.")


def export(pipeline_path: str, output_path: str, zip_output: int = 0, train_split_size: float = 0.75) -> bool:
    """Exports the data stored in a pipeline.

    Args:
        pipeline_path: Directory where the pipeline is stored.
        output_path: The path to store the dataset.
        train_split_size: The size of the trainig data.

    Returns:
        bool indicating success.
    """
    pipe = Pipeline(base_dir=pipeline_path)
    video_list: List[Dict[str, Any]] = []
    for source in pipe.video_item_dict:
        for video_id, item in pipe.video_item_dict[source].items():
            video_item = VideoItem(**pipe.video_item_dict[source][video_id])
            video_dir = join(pipeline_path, source, video_id)
            succ_metadata, video_metadata = load_metadata(video_dir=video_dir)
            if not succ_metadata:
                logger.info(f"Was not able to load the metadata of {video_dir}")
                continue
            succ_data, data_dict = perpare_data(
                video_dir=video_dir, source=source, video_id=video_id, video_item=video_item, metadata=video_metadata
            )
            if not succ_data:
                continue
            video_list.append(data_dict)

    logger.info(f"Found {len(video_list)} videos.")

    # Creating split
    random.shuffle(video_list)
    train_number = int(len(video_list) * train_split_size)
    train_list = video_list[:train_number]
    test_list = video_list[train_number:]

    # To keep an alphabetic ordering
    sorted(train_list, key=lambda x: x["identifier"])
    sorted(test_list, key=lambda x: x["identifier"])

    save_frames_to_disk(data_list=train_list, output_path=output_path, split="train")
    save_frames_to_disk(data_list=test_list, output_path=output_path, split="test")

    return True

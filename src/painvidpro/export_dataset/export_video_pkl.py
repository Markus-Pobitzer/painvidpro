"""Exports the pipleine to a hugging face dataset, an entry containing all frames, and meta data."""

import csv
import logging
import pickle
import random
from io import BytesIO
from os.path import join
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

from painvidpro.export_dataset.export_to_hf import prepare_data
from painvidpro.pipeline.input_file_format import VideoItem
from painvidpro.pipeline.pipeline import Pipeline
from painvidpro.utils.metadata import load_metadata


logger = logging.getLogger(__name__)


def prepare_pickel_dataset(video: Dict[str, Any], max_num_frames: int = -1) -> Tuple[List[bytes], List[float]]:
    """
    Processes and compresses frames from a video.

    Args:
        video (Dict[str, Any]): Video data dictionary.
        max_num_frames (int): Maximum number of frames to include.

    Returns:
        Tuple[List[bytes], List[float]]: Compressed frames and their progress values.
    """
    video_dir = video["video_dir"]
    video_path = Path(video_dir)

    # Process all frames for this video
    frames = video["extracted_frames"]
    start_frame = video["start_frame_idx"]
    end_frame = video["end_frame_idx"]

    frame_data: List[bytes] = []
    frame_progress_list: List[float] = []
    for frame_dict in frames:
        frame_idx = frame_dict.get("index", -1)
        frame_rel_path = frame_dict.get("path", "")
        if frame_idx < 0 or frame_rel_path == "":
            logger.info((f"Was not able to save frame {frame_dict} from" f" video dir {video_dir}."))
            continue

        frame_path = video_path / frame_rel_path
        try:
            with Image.open(frame_path) as img:
                # Compress image to bytes using JPEG format
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                compressed_frame = buffer.getvalue()
        except Exception as e:
            logger.error(f"Error loading frame {str(frame_path)}: {e}")
            continue
        progress = max(0.0, min(frame_idx / (end_frame - start_frame), 1.0))
        frame_data.append(compressed_frame)
        frame_progress_list.append(progress)

    if max_num_frames > 0 and len(frame_data) > max_num_frames:
        idx = np.round(np.linspace(0, len(frame_data) - 1, max_num_frames)).astype(int)
        frame_data = [frame_data[i] for i in idx]
        frame_progress_list = [frame_progress_list[i] for i in idx]
    return frame_data, frame_progress_list


def export_to_directory(
    video_list: List[Dict[str, Any]], out_dir: str, max_num_frames: int = -1, disable_tqdm: bool = False
) -> None:
    """
    Exports video data and metadata to a directory structure.

    Args:
        video_list (List[Dict[str, Any]]): List of video data dictionaries.
        out_dir (str): Output directory path.
        max_num_frames (int): Maximum number of frames to include.
    """
    base_dir = Path(out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = base_dir / "metadata.csv"

    with open(metadata_path, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["source", "video_url", "video_title", "art_style", "art_genre", "art_media"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for video in tqdm(video_list, desc=f"Saving to {out_dir}", disable=disable_tqdm):
            source = video["source"]
            video_url = video["video_url"]
            video_title = video["video_title"]
            art_style = ";".join(video["art_style"])
            art_genre = ";".join(video["art_genre"])
            art_media = ";".join(video["art_media"])

            video_out_dir = base_dir / source / video_url
            video_out_dir.mkdir(parents=True, exist_ok=True)

            try:
                ref_frame_path = join(video["video_dir"], video["reference_frame_name"])
                with Image.open(ref_frame_path) as reference_frame:
                    reference_frame = reference_frame.save(join(video_out_dir, "reference_frame.png"))

            except Exception as e:
                logger.error(f"Error loading reference frame: {e}")
                continue

            frame_data, frame_progress = prepare_pickel_dataset(video, max_num_frames=max_num_frames)
            if len(frame_data) <= 1:
                logger.error(f"Only found {len(frame_data)} frames in {video_url}, that is not enough!")
            # Save compressed frames
            with open(join(video_out_dir, "frame_data.pkl"), "wb") as f:
                pickle.dump(frame_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save progress values separately
            with open(join(video_out_dir, "frame_progress.pkl"), "wb") as f:
                pickle.dump(frame_progress, f, protocol=pickle.HIGHEST_PROTOCOL)

            writer.writerow(
                {
                    "source": source,
                    "video_url": video_url,
                    "video_title": video_title,
                    "art_style": art_style,
                    "art_genre": art_genre,
                    "art_media": art_media,
                }
            )


def export_video_to_pkl(
    pipeline_path: str,
    output_dir: str,
    train_split_size: float = 0.9,
    max_num_frames: int = -1,
    disable_tqdm: bool = False,
) -> None:
    """Exports pipeline data directly into a Hugging Face DatasetDict.

    Args:
        pipeline_path: Directory where the pipeline is stored.
        output_dir: Driectory to store the dataset.
        train_split_size: The size of the trainig data.
        sample_ref_frame_variation: If set, randomly selects a reference frame
            from the reference_frame_variations. If reference_frame_variations
            is empty, falls back to reference_frame_name.
        max_num_frames: If set to a vlue greater than 0, takes at most max_num_frames.

    Returns:
        A DatasetDicts.
    """
    pipe = Pipeline(base_dir=pipeline_path)
    video_list: List[Dict[str, Any]] = []

    # Process videos and collect metadata
    for source in pipe.video_item_dict:
        for video_id, item in pipe.video_item_dict[source].items():
            video_item = VideoItem(**item)
            video_dir = join(pipeline_path, source, video_id)
            succ_metadata, video_metadata = load_metadata(video_dir=video_dir)
            if not succ_metadata:
                logger.info(f"Was not able to load the metadata of {video_dir}")
                continue
            succ_data, data_dict = prepare_data(
                video_dir=video_dir, source=source, video_id=video_id, video_item=video_item, metadata=video_metadata
            )
            if not succ_data:
                continue
            video_list.append(data_dict)

    logger.info(f"Found {len(video_list)} videos.")

    # Split dataset
    random.shuffle(video_list)
    split_idx = int(len(video_list) * train_split_size)
    train_data = sorted(video_list[:split_idx], key=lambda x: x["identifier"])
    test_data = sorted(video_list[split_idx:], key=lambda x: x["identifier"])
    train_dir = join(output_dir, "train")
    test_dir = join(output_dir, "test")
    export_to_directory(
        video_list=train_data, out_dir=train_dir, max_num_frames=max_num_frames, disable_tqdm=disable_tqdm
    )
    export_to_directory(
        video_list=test_data, out_dir=test_dir, max_num_frames=max_num_frames, disable_tqdm=disable_tqdm
    )

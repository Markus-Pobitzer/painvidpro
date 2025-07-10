"""Exports the pipleine to a hugging face dataset, an entry containing all frames, and meta data."""

import logging
import random
from os.path import join
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image

from painvidpro.export_dataset.export_to_hf import _get_dataset_info, prepare_data
from painvidpro.pipeline.input_file_format import VideoItem
from painvidpro.pipeline.pipeline import Pipeline
from painvidpro.utils.metadata import load_metadata


logger = logging.getLogger(__name__)


def _get_features() -> Features:
    """Returns the Features of the dataset."""
    return Features(
        {
            "source": Value("string"),
            "video_url": Value("string"),
            "video_title": Value("string"),
            "art_style": Sequence(Value("string")),
            "art_genre": Sequence(Value("string")),
            "art_media": Sequence(Value("string")),
            "reference_frame": ImageFeature(),
            "frame_list": Sequence(ImageFeature()),
            "frame_progress_list": Sequence(Value("float32")),
        }
    )


def generate_examples(
    data_list: List[Dict[str, Any]], max_num_frames: int = -1
) -> Generator[Dict[str, Any], None, None]:
    """
    Generates examples by loading images directly from video directories.

    Args:
        data_list: List with the entries as Dicts.
        max_num_frames: If set to a vlue greater than 0, takes at most max_num_frames.

    Yields:
        Dict[str, Any]: A dictionary containing:
            - source: The video source
            - video_url: The video URL related to source
            - video_title: The title of the video
            - art_style: Sequence of art style
            - art_genre: Sequence of art genre
            - art_media: Sequence of art media
            - reference_frame: The reference frame image
            - frame_list: The frames from the video
            - frame_progress_list: Progress corresponding to each frame (entry in 0.0 to 1.0)
    """
    for video in data_list:
        video_dir = video["video_dir"]
        video_path = Path(video_dir)

        try:
            ref_frame_rel_path = video["reference_frame_name"]
            reference_frame = Image.open(join(video_dir, ref_frame_rel_path))
        except Exception as e:
            logger.error(f"Error loading reference frame: {e}")
            continue

        # Process all frames for this video
        frames = video["extracted_frames"]
        start_frame = video["start_frame_idx"]
        end_frame = video["end_frame_idx"]

        frame_list: List[Image.Image] = []
        frame_progress_list: List[float] = []
        for frame_dict in frames:
            frame_idx = frame_dict.get("index", -1)
            frame_rel_path = frame_dict.get("path", "")
            if frame_idx < 0 or frame_rel_path == "":
                logger.info((f"Was not able to save frame {frame_dict} from" f" video dir {video_dir}."))
                continue

            frame_path = video_path / frame_rel_path
            try:
                frame_img = Image.open(frame_path)
            except Exception as e:
                logger.error(f"Error loading frame {str(frame_path)}: {e}")
                continue
            progress = max(0.0, min(frame_idx / (end_frame - start_frame), 1.0))
            frame_list.append(frame_img)
            frame_progress_list.append(progress)

        if max_num_frames > 0 and len(frame_list) > max_num_frames:
            idx = np.round(np.linspace(0, len(frame_list) - 1, max_num_frames)).astype(int)
            frame_list = [frame_list[i] for i in idx]
            frame_progress_list = [frame_progress_list[i] for i in idx]
        yield {
            "source": video["source"],
            "video_url": video["video_url"],
            "video_title": video["video_title"],
            "art_style": video["art_style"],
            "art_genre": video["art_genre"],
            "art_media": video["art_media"],
            "reference_frame": reference_frame,
            "frame_list": frame_list,
            "frame_progress_list": frame_progress_list,
        }


def export_video_to_hf_dataset(
    pipeline_path: str,
    train_split_size: float = 0.9,
    sample_ref_frame_variation: bool = False,
    max_num_frames: int = -1,
) -> DatasetDict:
    """Exports pipeline data directly into a Hugging Face DatasetDict.

    Args:
        pipeline_path: Directory where the pipeline is stored.
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

    # Define dataset features
    features = _get_features()

    return DatasetDict(
        {
            "train": Dataset.from_generator(
                generate_examples,
                gen_kwargs={"data_list": train_data, "max_num_frames": max_num_frames},
                features=features,
                split="train",
                info=_get_dataset_info(),
            ),
            "test": Dataset.from_generator(
                generate_examples,
                gen_kwargs={"data_list": test_data, "max_num_frames": max_num_frames},
                features=features,
                split="test",
                info=_get_dataset_info(),
            ),
        }
    )

"""Exports the pipleine to a hugging face dataset, each video a seperate entry."""

import logging
import random
from os.path import join
from pathlib import Path
from typing import Any, Dict, Generator, List

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
        }
    )


def _get_features_data() -> Features:
    """Returns the Features of the data in the dataset."""
    return Features(
        {
            "video_url": Value("string"),
            "frame": ImageFeature(),
            "frame_progress": Value("float32"),
        }
    )


def generate_examples(
    data_list: List[Dict[str, Any]], sample_ref_frame_variation: bool = False
) -> Generator[Dict[str, Any], None, None]:
    """
    Generates examples by loading images directly from video directories.

    Args:
        data_list: List with the entries as Dicts.
        sample_ref_frame_variation: If set, randomly selects a reference frame
            from the reference_frame_variations. If reference_frame_variations
            is empty, falls back to reference_frame_name.

    Yields:
        Dict[str, Any]: A dictionary containing:
            - source: The video source
            - video_url: The video URL related to source
            - video_title: The title of the video
            - art_style: Sequence of art style
            - art_genre: Sequence of art genre
            - art_media: Sequence of art media
            - reference_frame: The reference frame image
    """
    for video in data_list:
        video_dir = video["video_dir"]

        try:
            ref_frame_rel_path = video["reference_frame_name"]
            if sample_ref_frame_variation and len(video["reference_frame_variations"]) > 0:
                ref_frame_rel_path = random.sample(video["reference_frame_variations"], 1)[0]["path"]
            reference_frame = Image.open(join(video_dir, ref_frame_rel_path))
        except Exception as e:
            logger.error(f"Error loading reference frame: {e}")
            continue

        yield {
            "source": video["source"],
            "video_url": video["video_url"],
            "video_title": video["video_title"],
            "art_style": video["art_style"],
            "art_genre": video["art_genre"],
            "art_media": video["art_media"],
            "reference_frame": reference_frame,
        }


def generate_data_examples(data_list: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Generates examples by loading images directly from video directories.

    Args:
        data_list: List with the entries as Dicts.

    Yields:
        Dict[str, Any]: A dictionary containing:
            - video_url: The url of the video
            - frame: The current frame image
            - frame_progress: Progress through the video (0.0 to 1.0)
    """
    for video in data_list:
        video_dir = video["video_dir"]
        video_path = Path(video_dir)

        # Process all frames for this video
        frames = video["extracted_frames"]
        start_frame = video["start_frame_idx"]
        end_frame = video["end_frame_idx"]

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
            yield {
                "video_url": video["video_url"],
                "frame": frame_img,
                "frame_progress": progress,
            }


def export_video_as_entry_to_hf_dataset(
    pipeline_path: str, train_split_size: float = 0.9, sample_ref_frame_variation: bool = False
) -> Dict[str, DatasetDict]:
    """Exports pipeline data directly into a Hugging Face DatasetDict.

    Args:
        pipeline_path: Directory where the pipeline is stored.
        train_split_size: The size of the trainig data.
        sample_ref_frame_variation: If set, randomly selects a reference frame
            from the reference_frame_variations. If reference_frame_variations
            is empty, falls back to reference_frame_name.

    Returns:
        A dict containing DatasetDicts.
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
    data_features = _get_features_data()

    return {
        "splits": DatasetDict(
            {
                "train": Dataset.from_generator(
                    generate_examples,
                    gen_kwargs={"data_list": train_data, "sample_ref_frame_variation": sample_ref_frame_variation},
                    features=features,
                    split="train",
                    info=_get_dataset_info(),
                ),
                "test": Dataset.from_generator(
                    generate_examples,
                    gen_kwargs={"data_list": test_data, "sample_ref_frame_variation": sample_ref_frame_variation},
                    features=features,
                    split="test",
                    info=_get_dataset_info(),
                ),
            }
        ),
        "data": DatasetDict(
            {
                "train_data": Dataset.from_generator(
                    generate_data_examples,
                    gen_kwargs={"data_list": train_data},
                    features=data_features,
                    split="train",
                    info=_get_dataset_info(),
                ),
                "test_data": Dataset.from_generator(
                    generate_data_examples,
                    gen_kwargs={"data_list": test_data},
                    features=data_features,
                    split="test",
                    info=_get_dataset_info(),
                ),
            }
        ),
    }

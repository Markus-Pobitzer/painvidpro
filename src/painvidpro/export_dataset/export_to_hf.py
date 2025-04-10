"""Exports the pipleine to a hugging face dataset."""

import logging
import random
from os.path import join
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from datasets import Dataset, DatasetDict, DatasetInfo, Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image

from painvidpro.pipeline.input_file_format import VideoItem
from painvidpro.pipeline.pipeline import Pipeline
from painvidpro.utils.metadata import load_metadata


logger = logging.getLogger(__name__)


def prepare_data(
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
        "video_title": metadata.get("title", ""),
        "video_license": metadata.get("license", ""),
        "video_channel": metadata.get("channel", ""),
        "video_channel_id": metadata.get("video_channel_id", ""),
        "art_style": metadata.get("art_style", []),
        "art_genre": metadata.get("art_genre", []),
        "art_media": metadata.get("art_media", []),
    }
    return True, ret


def _get_features() -> Features:
    """Returns the Features of the dataset."""
    return Features(
        {
            "source": Value("string"),
            "video_url": Value("string"),
            "video_title": Value("string"),
            "frame": ImageFeature(),
            "art_style": Sequence(Value("string")),
            "art_genre": Sequence(Value("string")),
            "art_media": Sequence(Value("string")),
            "reference_frame": ImageFeature(),
            "frame_progress": Value("float32"),
        }
    )


def _get_dataset_info() -> DatasetInfo:
    """Returns the dataset info."""
    dataset_info = DatasetInfo(
        description="PainVidPro: Painting Video Processing Dataset.",
        citation=(
            "@misc{PainVidPro2025,\n"
            "  author = {Markus Pobitzer},\n"
            "  title = {PainVidPro Dataset},\n"
            "  year = {2025},\n"
            "  howpublished = {\\url{https://github.com/Markus-Pobitzer/painvidpro}},\n"
            "}"
        ),
        homepage="https://github.com/Markus-Pobitzer/painvidpro",
    )
    return dataset_info


def generate_examples(data_list: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
    """
    Generates examples by loading images directly from video directories.

    Args:
        data_list: List with the entries as Dicts.

    Yields:
        Dict[str, Any]: A dictionary containing:
            - source: The video source
            - video_url: The video URL related to source
            - video_title: The title of the video
            - frame_progress: Progress through the video (0.0 to 1.0)
            - prompt: Text prompt associated with the frame
            - frame: The current frame image
            - art_style: Sequence of art style
            - art_genre: Sequence of art genre
            - art_media: Sequence of art media
            - reference_frame: The reference frame image
    """
    for video in data_list:
        video_dir = video["video_dir"]
        video_path = Path(video_dir)
        try:
            # Load reference frame once per video
            reference_frame = Image.open(join(video_dir, video["reference_frame_name"]))
        except Exception as e:
            logger.error(f"Error loading reference frame: {e}")
            continue

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
                "source": video["source"],
                "video_url": video["video_url"],
                "video_title": video["video_title"],
                "frame": frame_img,
                "art_style": video["art_style"],
                "art_genre": video["art_genre"],
                "art_media": video["art_media"],
                "reference_frame": reference_frame,
                "frame_progress": progress,
            }


def export_to_hf_dataset(pipeline_path: str, train_split_size: float = 0.9) -> DatasetDict:
    """Exports pipeline data directly into a Hugging Face DatasetDict.

    Args:
        pipeline_path: Directory where the pipeline is stored.
        train_split_size: The size of the trainig data.

    Returns:
        A DatasetDict.
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
                gen_kwargs={"data_list": train_data},
                features=features,
                split="train",
                info=_get_dataset_info(),
            ),
            "test": Dataset.from_generator(
                generate_examples,
                gen_kwargs={"data_list": test_data},
                features=features,
                split="test",
                info=_get_dataset_info(),
            ),
        }
    )


def save_sharded_dataset_dict(
    ds_dict: DatasetDict, output_dir: str, max_shard_size: str = "500MB", num_proc: Optional[int] = None
) -> None:
    """Save each split of the DatasetDict in sharded Parquet files.

    Each shard is smaller than the specified size.

    Args:
        ds_dict: The Hugging Face DatasetDict to be saved.
        output_dir: The directory where the Parquet files will be saved.
        max_shard_size: The maximum size of each shard as string.
        num_proc: If set uses number processors for multi processing.
    """
    ds_dict.save_to_disk(dataset_dict_path=output_dir, max_shard_size=max_shard_size, num_proc=num_proc)

"""Exports, i.e. mainly copies the h5 file in a shared dataset."""

import json
import logging
import random
import shutil
from os.path import join
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from painvidpro.data_storage.hdf5_video_archive import DynamicVideoArchive
from painvidpro.pipeline.pipeline import Pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_pipeline(
    pipeline_path: str,
    output_dir: str,
    train_split_size: float = 0.8,
    eval_split_size: float = 0.1,
    video_source_split: Optional[str] = "video_sources/video_sources_split.json",
    entries_in_video_source_split: bool = False,
    seed: Optional[int] = 42,
    frame_data_file="frame_data.h5",
    disable_tqdm: bool = False,
) -> None:
    """Processes video pipeline data and copies it into a split directory structure (train/test).

    Args:
        pipeline_path: Directory where the source pipeline is stored.
        output_dir: Target directory to store the exported video files.
        train_split_size: Float between 0 and 1 representing the proportion of training data.
        video_source_split: Path to a JSON file defining specific splits.
        entries_in_video_source_split: If True, only videos found in the split file are exported.
        seed: Random seed for reproducible splitting.
        frame_data_file: The name of the HDF5 file containing frame metadata.
        disable_tqdm: If set disables tqdm progress bar.
    """
    if train_split_size + eval_split_size > 1.0:
        raise ValueError("Train and Eval splits cannot sum to more than 1.0")

    pipe = Pipeline(base_dir=pipeline_path)
    video_dir_list: List[str] = []

    if seed is not None:
        random.seed(seed)

    split_dict: Dict[str, Any] = {}
    if video_source_split is not None:
        # Check if entry is in file
        with open(video_source_split) as f:
            split_dict = json.load(f)

    # Process videos and collect metadata
    for source in pipe.video_item_dict:
        for video_id, _ in tqdm(
            pipe.video_item_dict[source].items(), f"Exporting from {source}", disable=disable_tqdm
        ):
            video_dir = join(pipeline_path, source, video_id)
            with DynamicVideoArchive(join(video_dir, frame_data_file), mode="r") as frame_dataset:
                metadata = frame_dataset.get_global_metadata()
                if metadata.get("exclude_video", False):
                    logger.info(f"Excluding {video_dir}, the exclude_video is set in the metadata.")
                    continue
                if len(frame_dataset) == 0:
                    logger.info(f"Excluding {video_dir}, no extracted frames were found.")
                    continue

            if split_dict and source in split_dict and video_id in split_dict[source]:
                split = split_dict[source][video_id]
            else:
                if entries_in_video_source_split:
                    # Entry was ot found
                    logger.info(
                        (
                            f"Excluding {video_dir}, it was not found in the video_source_split {video_source_split} and entries_in_video_source_split was set."
                        )
                    )
                    continue

                split = "test"
                r = random.random()
                if r <= train_split_size:
                    split = "train"
                elif r <= (train_split_size + eval_split_size):
                    split = "eval"

            out_dir = join(output_dir, split, source, video_id)
            try:
                shutil.copytree(video_dir, out_dir)
            except Exception as e:
                logger.info((f"Was not able to copy {video_dir} to {out_dir}, skipping entry due to error: {e}"))
                continue

            video_dir_list.append(video_dir)

    logger.info(f"Exported {len(video_dir_list)} videos to {output_dir}.")
    print(f"Exported {len(video_dir_list)} videos to {output_dir}.")

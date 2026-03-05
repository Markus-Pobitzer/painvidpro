"""Exports, i.e. mainly copies the h5 file in a shared dataset."""

import json
import logging
import random
import shutil
from os.path import join
from typing import Any, Dict, List, Optional, Tuple

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


def export_pipeline_n_fold(
    pipeline_path: str,
    output_dir: str,
    n_splits: int = 5,
    seed: Optional[int] = 42,
    frame_data_file: str = "frame_data.h5",
    disable_tqdm: bool = False,
) -> None:
    """Processes video pipeline data and exports it into N cross-validation folds.

    Args:
        pipeline_path: Directory where the source pipeline is stored.
        output_dir: Target directory to store the exported fold directories.
        n_splits: Number of cross-validation folds to create (default is 5).
        seed: Random seed for reproducible splitting.
        frame_data_file: The name of the HDF5 file containing frame metadata.
        disable_tqdm: If set disables tqdm progress bar.
    """
    pipe = Pipeline(base_dir=pipeline_path)

    if seed is not None:
        random.seed(seed)

    # Collect all valid videos grouped by source to ensure stratified splits
    # Dict structure: {source: [(video_id, video_dir), ...]}
    valid_videos_by_source: Dict[str, List[Tuple[str, str]]] = {source: [] for source in pipe.video_item_dict}

    for source in pipe.video_item_dict:
        for video_id, _ in tqdm(
            pipe.video_item_dict[source].items(), desc=f"Validating videos from {source}", disable=disable_tqdm
        ):
            video_dir = join(pipeline_path, source, video_id)

            with DynamicVideoArchive(join(video_dir, frame_data_file), mode="r") as frame_dataset:
                metadata = frame_dataset.get_global_metadata()
                if metadata.get("exclude_video", False):
                    logger.info(f"Excluding {video_dir}, exclude_video is set in metadata.")
                    continue
                if len(frame_dataset) == 0:
                    logger.info(f"Excluding {video_dir}, no extracted frames found.")
                    continue

            valid_videos_by_source[source].append((video_id, video_dir))

    # Partition valid videos into N folds (stratified by source)
    folds: List[List[Tuple[str, str, str]]] = [[] for _ in range(n_splits)]

    for source, videos in valid_videos_by_source.items():
        # Shuffle videos within the source to ensure random distribution
        random.shuffle(videos)

        # Distribute videos round-robin into the N folds
        for i, (video_id, video_dir) in enumerate(videos):
            folds[i % n_splits].append((source, video_id, video_dir))

    # Export the N folds
    total_exported = 0
    for fold_idx in range(n_splits):
        fold_out_dir = join(output_dir, f"fold_{fold_idx}")
        logger.info(f"Exporting Fold {fold_idx + 1}/{n_splits} to {fold_out_dir} ...")

        for chunk_idx, chunk_videos in enumerate(folds):
            # If the chunk matches the current fold index, it's the test set.
            # Otherwise, it belongs to the training set.
            split_name = "test" if chunk_idx == fold_idx else "train"

            for source, video_id, video_dir in tqdm(
                chunk_videos, desc=f"Copying Fold {fold_idx} {split_name}", disable=disable_tqdm, leave=False
            ):
                out_dir = join(fold_out_dir, split_name, source, video_id)
                try:
                    shutil.copytree(video_dir, out_dir)
                    total_exported += 1
                except Exception as e:
                    logger.info(f"Failed to copy {video_dir} to {out_dir}, skipping. Error: {e}")

    # total_exported represents the total directory copies made.
    # To get unique videos exported, it's total_exported / n_splits
    unique_videos = total_exported // n_splits
    logger.info(f"Successfully exported {unique_videos} unique videos across {n_splits} folds into {output_dir}.")
    print(f"Exported {unique_videos} videos into {n_splits} folds at {output_dir}.")

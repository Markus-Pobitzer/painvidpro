"""Utility functions for dataset processing."""

from typing import List

from datasets import DatasetDict, concatenate_datasets


def combine_dataset_dicts(dataset_dicts: List[DatasetDict]) -> DatasetDict:
    """Combine multiple DatasetDict objects into one."""
    combined = DatasetDict()

    # Get all possible splits across all dataset_dicts
    all_splits = set()
    for dd in dataset_dicts:
        all_splits.update(dd.keys())

    for split in all_splits:
        # Filter dataset_dicts that contain this split
        split_datasets = [dd[split] for dd in dataset_dicts if split in dd]

        if split_datasets:
            combined_split = concatenate_datasets(split_datasets)
            combined[split] = combined_split

    return combined

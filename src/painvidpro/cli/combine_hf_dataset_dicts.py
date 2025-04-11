"""Combines several HuggingFace dataset dicts into one."""

import argparse
import os
from typing import List

from datasets import DatasetDict, concatenate_datasets, load_from_disk


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


def main():
    parser = argparse.ArgumentParser(description="Combine multiple Hugging Face DatasetDicts and save the result")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to input DatasetDict directories")
    parser.add_argument("--output_dir", required=True, help="Output directory for the combined DatasetDict")
    parser.add_argument("--max_shard_size", default="5GB", help="Maximum shard size when saving (default: 5GB)")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=os.cpu_count(),
        help=f"Number of processes to use (default: {os.cpu_count()} - all CPUs)",
    )

    args = parser.parse_args()

    # Load all input DatasetDicts
    dataset_dicts = []
    print("\n🚀 Loading datasets...")
    for input_path in args.inputs:
        dataset_dicts.append(load_from_disk(input_path))
    print("✅ Datasets loaded successfully")

    print("\n🚀 Combining datasets...")
    # Combine datasets
    combined = combine_dataset_dicts(dataset_dicts)
    print("✅ Datasets combined successfully")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n🚀 Saving combined dataset...")
    # Save combined dataset
    combined.save_to_disk(args.output_dir, max_shard_size=args.max_shard_size, num_proc=args.num_proc)
    print(f"✅ Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()

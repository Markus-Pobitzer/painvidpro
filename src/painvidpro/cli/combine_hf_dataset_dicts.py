"""Combines several HuggingFace dataset dicts into one."""

import argparse
import os

from datasets import load_from_disk

from painvidpro.export_dataset.utils import combine_dataset_dicts


def main():
    parser = argparse.ArgumentParser(description="Combine multiple Hugging Face DatasetDicts and save the result")
    parser.add_argument("--inputs", nargs="+", required=True, help="Paths to input DatasetDict directories")
    parser.add_argument("--output_dir", required=True, help="Output directory for the combined DatasetDict")
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
    combined.save_to_disk(
        args.output_dir,
        num_proc=args.num_proc,
        num_shards={
            "train": 20,
            "test": 2,
        },
    )
    print(f"✅ Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Combines several HuggingFace dataset dicts containing video as entry into one."""

import argparse
import os

from datasets import load_from_disk

from painvidpro.export_dataset.utils import combine_dataset_dicts


def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple custom Hugging Face DatasetDicts and save the result"
    )
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
    dataset_split_dicts = []
    dataset_data_dicts = []
    print("\n🚀 Loading datasets...")
    for input_path in args.inputs:
        dataset_split_dicts.append(load_from_disk(os.path.join(input_path, "splits")))
        dataset_data_dicts.append(load_from_disk(os.path.join(input_path, "data")))
    print("✅ Datasets loaded successfully")

    print("\n🚀 Combining datasets...")
    # Combine datasets
    combined_split = combine_dataset_dicts(dataset_split_dicts)
    combined_data = combine_dataset_dicts(dataset_data_dicts)
    print("✅ Datasets combined successfully")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n🚀 Saving combined dataset...")
    # Save combined dataset
    combined_split.save_to_disk(
        dataset_dict_path=os.path.join(args.output_dir, "splits"),
        num_shards={
            "train": 2,
            "test": 2,
        },  # 2 shards needed then it is easier to load
        num_proc=args.num_proc,
    )
    combined_data.save_to_disk(
        dataset_dict_path=os.path.join(args.output_dir, "data"),
        num_shards={
            "train_data": 20,
            "test_data": 10,
        },
        num_proc=args.num_proc,
    )
    print(f"✅ Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()

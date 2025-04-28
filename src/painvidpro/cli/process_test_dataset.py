"""Processes a split of the dataset, i.e. resizing images."""

import argparse
import os

from datasets import DatasetDict, load_dataset

from painvidpro.export_dataset.process_test_dataset import apply_img_processing


map_function_list = ["resize_and_center_crop", "resize"]


def main():
    parser = argparse.ArgumentParser(description="Processes a Dataset split, i.e. resizes, cropping.")
    parser.add_argument("--dataset_name", required=True, help="Hugging Face dataset.")
    parser.add_argument("--output_dir", required=True, help="Output directory for the combined Dataset")
    parser.add_argument("--map_function", default="resize_and_center_crop", help=f"One from {map_function_list}")
    parser.add_argument("--width", default=512, help="Image width")
    parser.add_argument("--height", default=512, help="Image height")
    parser.add_argument("--dataset_split", default="test", help="The Split to select.")
    parser.add_argument("--max_shard_size", default="5GB", help="Maximum shard size when saving (default: 5GB)")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=os.cpu_count(),
        help=f"Number of processes to use (default: {os.cpu_count()} - all CPUs)",
    )

    args = parser.parse_args()

    dataset = load_dataset(
        args.dataset_name,
        split=args.dataset_split,
    )

    print(f"\n🚀 Processing datasets with {args.map_function} map fucntion.")
    # Combine datasets
    dataset = apply_img_processing(ds=dataset, config=vars(args))
    print("✅ Datasets processed successfully")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n🚀 Saving processed dataset...")
    # Save dataset as a DatasetDict since it is can be loaded with laod_dataset
    ds_dict = DatasetDict({args.dataset_split: dataset})
    ds_dict.save_to_disk(args.output_dir, max_shard_size=args.max_shard_size, num_proc=args.num_proc)
    print(f"✅ Dataset saved to {args.output_dir}")


if __name__ == "__main__":
    main()

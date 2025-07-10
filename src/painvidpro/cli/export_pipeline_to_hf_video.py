"""Creates a hugging face dataset from a pipeline."""

import argparse
import os
import sys

from painvidpro.export_dataset.export_video import export_video_to_hf_dataset


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for dataset processing.

    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Load pipeline and convert to Hugging Face format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument("pipeline_path", type=str, help="Directory containing the pipeline.")
    parser.add_argument("output_dir", type=str, help="Output directory for processed Hugging Face dataset")
    parser.add_argument("--train_split_size", type=float, default=0.9, help="The training split size, between [0, 1].")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=-1,
        help="Number of processes to use for parallelism, default uses available resources.",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=-1,
        help=("If set to a vlue greater than 0, takes at most max_num_frames."),
    )

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate input arguments and ensure system requirements are met.

    Args:
        args: Parsed command line arguments

    Raises:
        ValueError: If any arguments are invalid
        FileNotFoundError: If input directory doesn't exist
    """
    if not os.path.isdir(args.pipeline_path):
        raise FileNotFoundError(f"Directory containg pipeline not found: {args.pipeline_path}")


def main() -> None:
    """Main processing pipeline for dataset conversion."""
    try:
        args = parse_arguments()
        validate_arguments(args)

        # Step 1: Create Hugging Face DatasetDict
        print("\n🚀 Creating Hugging Face dataset...")
        datasetdict = export_video_to_hf_dataset(
            pipeline_path=args.pipeline_path,
            train_split_size=args.train_split_size,
            max_num_frames=args.max_num_frames,
        )
        print("✅ Dataset created successfully")

        num_proc = int(args.num_proc)
        # If negative value use 0.8 of available cpus
        if num_proc < 1:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                num_proc = int(cpu_count * 0.8)
            else:
                num_proc = 1

        # Step 2: Save sharded dataset
        print("\n🚀 Saving sharded dataset...")
        datasetdict.save_to_disk(
            dataset_dict_path=args.output_dir,
            num_shards={
                "train": 10,
                "test": 2,
            },
            num_proc=num_proc,
        )
        print(f"✅ Dataset saved to {args.output_dir}")

    except Exception as e:
        print(f"\n❌ Error processing dataset: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

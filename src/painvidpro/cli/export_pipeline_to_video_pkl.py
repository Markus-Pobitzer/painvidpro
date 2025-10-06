"""Creates a directory structure with pickeled frames."""

import argparse
import os
import sys

from painvidpro.export_dataset.export_video_pkl import export_video_to_pkl


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
        "--max_num_frames",
        type=int,
        default=-1,
        help=("If set to a vlue greater than 0, takes at most max_num_frames."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=("Seed when the dataset gets split into train and test."),
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
        print("\n🚀 Creating Pickel dataset ...")
        export_video_to_pkl(
            pipeline_path=args.pipeline_path,
            output_dir=args.output_dir,
            train_split_size=args.train_split_size,
            max_num_frames=args.max_num_frames,
            seed=args.seed,
        )
        print(f"✅ Dataset saved to {args.output_dir}")

    except Exception as e:
        print(f"\n❌ Error processing dataset: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

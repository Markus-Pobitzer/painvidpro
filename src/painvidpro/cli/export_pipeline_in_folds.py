"""Creates a directory structure with test and train split."""

import argparse
import os
import sys

from painvidpro.utils.export_dataset import export_pipeline_n_fold


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
    parser.add_argument("--number_of_folds", type=int, default=5, help="The number of folds to gerentate.")
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
        print(f"\n🚀 Creating dataset with {args.number_of_folds} folds ...")
        export_pipeline_n_fold(
            pipeline_path=args.pipeline_path,
            output_dir=args.output_dir,
            n_splits=args.number_of_folds,
            seed=args.seed,
        )
        print(f"✅ Dataset saved to {args.output_dir}")

    except Exception as e:
        print(f"\n❌ Error processing dataset: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Main file to export a Pipeline into an intermediate dataset."""

import argparse
import logging

from painvidpro.export_dataset.extracted_frames import export


logger = logging.getLogger(__name__)


class Main:
    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Export pipeline data.")
        parser.add_argument("pipeline_path", type=str, help="Directory where the pipeline is stored.")
        parser.add_argument("output_path", type=str, help="Path to store the dataset.")
        parser.add_argument(
            "--train_split_size", type=float, default=0.9, help="Training data split size (default: 0.9)."
        )
        args = parser.parse_args()

        success = export(
            pipeline_path=args.pipeline_path, output_path=args.output_path, train_split_size=args.train_split_size
        )
        if success:
            logger.info("Export succeeded.")
        else:
            logger.error("Export failed.")


if __name__ == "__main__":
    Main.main()

import argparse
import logging
from pathlib import Path
from typing import List

from painvidpro.pipeline.pipeline import Pipeline


def process_jsonl_files(base_dir: str, jsonl_files: List[str]) -> None:
    """Process multiple .jsonl files using the pipeline.

    Args:
        base_dir: Root directory for the pipeline data
        jsonl_files: List of paths to .jsonl files to process
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    try:
        # Initialize pipeline
        logger.info(f"Initializing pipeline with base directory: {base_dir}")
        pipeline = Pipeline(base_dir=base_dir)

        # Process each file
        for file_path in jsonl_files:
            try:
                path = Path(file_path)
                if not path.exists():
                    logger.error(f"File not found: {file_path}")
                    continue
                if path.suffix != ".jsonl":
                    logger.warning(f"Skipping non-JSONL file: {file_path}")
                    continue

                logger.info(f"Processing file: {path.resolve()}")
                processed_paths = pipeline.process_video_input(str(path))
                logger.info(f"Successfully processed {len(processed_paths)} video(s) from {path.name}")

            except Exception as e:
                logger.error(f"Failed to process {path.name}: {str(e)}", exc_info=True)
                continue

        logger.info("Finished processing all specified files")

    except Exception as e:
        logger.error(f"Critical error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple .jsonl files")
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory for the pipeline data")
    parser.add_argument("--files", type=str, required=True, nargs="+", help="Paths to .jsonl files to process")

    args = parser.parse_args()

    process_jsonl_files(base_dir=args.base_dir, jsonl_files=args.files)

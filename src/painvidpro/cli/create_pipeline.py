import argparse
import logging
from pathlib import Path
from typing import List

from painvidpro.pipeline.pipeline import Pipeline


def register_jsonl_files(base_dir: str, jsonl_files: List[str]) -> None:
    """Registers multiple .jsonl files using the pipeline.

    Args:
        base_dir: Root directory for the pipeline data.
        jsonl_files: List of paths to .jsonl files to register.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    logger = logging.getLogger(__name__)

    try:
        # Initialize pipeline
        logger.info(f"Initializing pipeline with base directory: {base_dir}")
        pipeline = Pipeline(base_dir=base_dir)

        # Register each file
        for file_path in jsonl_files:
            try:
                path = Path(file_path)
                if not path.exists():
                    logger.error(f"File not found: {file_path}")
                    continue
                if path.suffix != ".jsonl":
                    logger.warning(f"Skipping non-JSONL file: {file_path}")
                    continue

                logger.info(f"Registering file: {path.resolve()}")
                processed_paths = pipeline.register_video_input(str(path))
                logger.info(f"Successfully registered {len(processed_paths)} video(s) from {path.name}")

            except Exception as e:
                logger.error(f"Failed to register {path.name}: {str(e)}", exc_info=True)
                continue

        logger.info("Finished registering all specified files")

    except Exception as e:
        logger.error(f"Critical error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register multiple .jsonl files")
    parser.add_argument("--base-dir", type=str, required=True, help="Base directory for the pipeline data")
    parser.add_argument("--files", type=str, required=True, nargs="+", help="Paths to .jsonl files to register")

    args = parser.parse_args()

    register_jsonl_files(base_dir=args.base_dir, jsonl_files=args.files)

"""Processes entries of a pipeline with pecified procssor."""

import argparse
import json
import logging

from painvidpro.pipeline.pipeline import Pipeline


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process videos in the pipeline using specified processors.")
    parser.add_argument("--base_dir", required=True, help="Base directory for the pipeline")
    parser.add_argument(
        "--processor_config", required=True, help="Path to the JSONL file containing processor configurations"
    )
    args = parser.parse_args()

    pipeline = Pipeline(args.base_dir)

    with open(args.processor_config, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                config = json.loads(line)
                processors = config.get("video_processor", [])
                for proc in processors:
                    proc_name = proc.get("name")
                    proc_config = proc.get("config", {})
                    if proc_name:
                        logger.info(f"Applying processor '{proc_name}' from line {line_num}")
                        pipeline.process_overwrite_processor(
                            processor_name=proc_name,
                            processor_config=proc_config,
                        )
                    else:
                        logger.info(f"Warning: Processor entry in line {line_num} is missing a 'name' key")
            except json.JSONDecodeError as e:
                logger.info(f"Error decoding JSON on line {line_num}: {e}")


if __name__ == "__main__":
    main()

"""Augments the pipeline entries by creating reference frames."""

import argparse
import logging

from painvidpro.pipeline.pipeline import Pipeline


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Augments the pipeline entries by creating reference frames.")
    parser.add_argument("--base_dir", required=True, help="Base directory for the pipeline")
    parser.add_argument(
        "--processor_name", required=True, default="ProcessorRefFrameVariations", help="The processor to use."
    )
    parser.add_argument(
        "--remove_previous_ref_frames",
        action="store_true",
        help=("If set, removes the previous generated ref frames if any."),
    )
    parser.add_argument(
        "--remove_previous_ref_tags",
        action="store_true",
        help=("If set, removes the previous generated ref tags if any."),
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help=("Sets enable_sequential_cpu_offload=True in the processors config."),
    )
    args = parser.parse_args()

    pipeline = Pipeline(args.base_dir)

    if args.remove_previous_ref_frames:
        pipeline.remove_ref_frame_variations()

    if args.remove_previous_ref_tags:
        pipeline.remove_ref_frame_tags()

    processor_config = {}
    if args.enable_sequential_cpu_offload:
        processor_config["enable_sequential_cpu_offload"] = True
    pipeline.process_overwrite_processor(
        processor_name=args.processor_name,
        processor_config=processor_config,  # Using default
    )


if __name__ == "__main__":
    main()

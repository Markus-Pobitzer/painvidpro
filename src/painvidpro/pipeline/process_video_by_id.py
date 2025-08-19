import argparse

from painvidpro.logging.logging import setup_logger
from painvidpro.pipeline.pipeline import Pipeline


def main(base_dir: str, source: str, video_id: str) -> None:
    """Main function to process a specific video in the pipeline.

    Args:
        base_dir: Root directory for the pipeline data
        source: Video source platform (e.g., 'youtube')
        video_id: ID of the video to process
        batch_size: Optional batch size for processing
    """
    try:
        logger = setup_logger(__name__)
        logger.info(f"Initializing pipeline with base directory: {base_dir}")

        # Instantiate the pipeline
        pipeline = Pipeline(base_dir=base_dir)

        logger.info(f"Processing video {video_id} from source {source}")
        pipeline.process_video_by_id(source=source, video_id=video_id)

        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific video in the pipeline")
    parser.add_argument("--dir", type=str, required=True, help="Base directory for the pipeline data")
    parser.add_argument("--source", type=str, required=True, help="Source platform of the video (e.g., youtube)")
    parser.add_argument("--video_id", type=str, required=True, help="ID of the video to process")

    args = parser.parse_args()

    main(base_dir=args.dir, source=args.source, video_id=args.video_id)

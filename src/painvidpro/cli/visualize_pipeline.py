"""Visualize extracted content from a pipeline."""

import argparse

from painvidpro.visualization.extracted_gradio_app import gradio_app


def main():
    parser = argparse.ArgumentParser(description="Visualize extracted content from a pipeline.")
    parser.add_argument("pipeline_path", type=str, help="Directory containing the pipeline.")

    args = parser.parse_args()
    gradio_app(args.pipeline_path)


if __name__ == "__main__":
    main()

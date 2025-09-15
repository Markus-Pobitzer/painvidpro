"""The Pipeline to process video inputs."""

import copy
import dataclasses
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from painvidpro.logging.logging import setup_logger
from painvidpro.pipeline.input_file_format import VideoItem
from painvidpro.processors.factory import ProcessorsFactory
from painvidpro.utils.metadata import load_metadata, save_metadata
from painvidpro.utils.ref_frame_variations import clean_ref_frame_variations
from painvidpro.video_processing.youtube import get_info_from_yt_url


class Pipeline:
    def __init__(self, base_dir: str):
        self.logger = setup_logger(name=__name__)
        self.base_dir = Path(base_dir)
        self.youtube_dir = self.base_dir / "youtube"
        # Stores the video data
        self.video_item_dict: Dict[str, Dict[str, Dict[str, Any]]]
        self.save_file = "pipeline.json"
        self._ensure_dirs()
        self.load()

    def _ensure_dirs(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.youtube_dir.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        """Save pipeline state to JSON"""
        pipeline_path = self.base_dir / self.save_file
        with open(pipeline_path, "w") as f:
            json.dump({"video_item_dict": self.video_item_dict}, f, indent=4)

    def load(self) -> None:
        """Load pipeline from directory."""
        pipeline_path = Path(self.base_dir) / self.save_file
        if os.path.isfile(pipeline_path):
            with open(pipeline_path, "r") as f:
                data = json.load(f)
            self.video_item_dict = data["video_item_dict"]
        else:
            self.video_item_dict = {"youtube": {}}
            self.save()

    def register_youtube_video_input(self, video_item: VideoItem) -> Tuple[bool, List[str]]:
        """Registers a YouTube video input and updates the video item dictionary.

        Args:
            video_item: An instance of VideoItem containing details about the YouTube video.

        Returns:
            A tuple where the first element is a boolean indicating success, and the second
            element is a list of strings with the paths to the registered video directories
            or error messages.
        """
        url = video_item.url
        video_info_list = get_info_from_yt_url(url=url)
        ret: List[str] = []

        if len(video_info_list) == 0:
            return False, [f"Was not able to extract information from url {url}."]

        for video_data in video_info_list:
            video_id = video_data["id"]
            if video_id == "":
                self.logger.info(f"Empty video id for entry {video_data} with url {url}")
            elif video_id in self.video_item_dict["youtube"]:
                self.logger.info(f"Skipping video entry {video_data}, already in pipeline!")
            elif video_item.only_channel_vid and video_data["channel_id"] not in video_item.channel_ids:
                self.logger.info(f"Skipping video entry {video_data}, channel id not in channel_ids!")
            else:
                video_dir = self.youtube_dir / video_id
                video_dir.mkdir(parents=True, exist_ok=True)
                # Metadata is jsut a dict at the moment
                metadata = video_data
                metadata["art_style"] = video_item.art_style
                metadata["art_genre"] = video_item.art_genre
                metadata["art_media"] = video_item.art_media
                metadata["processed"] = False
                save_metadata(video_dir=video_dir, metadata=metadata)
                # Shallow copy of video_item to overwrite url
                entry = dataclasses.replace(video_item)
                entry.url = video_id
                self.video_item_dict["youtube"][video_id] = dataclasses.asdict(entry)
                ret.append(str(video_dir))
        return True, ret

    def register_video_input(self, video_input_file: str) -> List[str]:
        """Register a video input file and updates the video item dictionary.

        Args:
            video_input_file: The path to the file containing video item details in JSONL format.

        Returns:
            A list of strings with the paths to the registered video directories.
        """
        video_item_list: List[VideoItem] = []
        ret: List[str] = []
        with open(video_input_file) as f:
            for line in f:
                video_item_list.append(VideoItem(**json.loads(line)))

        for vi in video_item_list:
            if vi.source == "youtube":
                succ, msg_list = self.register_youtube_video_input(vi)
                if succ:
                    ret += msg_list
                else:
                    self.logger.info(
                        (
                            f"Was not successfull in registering Video Item {vi}."
                            f"The occurred problem: {msg_list[0]}."
                        )
                    )
            else:
                self.logger.info(f"No support for source {vi.source} in Video Item {vi}.")
        self.save()
        return ret

    def process_video(self, video_data: Dict[str, Any], batch_size: int = -1):
        """Applies the processor onto a video item."""
        source = video_data["source"]
        if source == "youtube":
            video_dir = self.youtube_dir / video_data["url"]
            for processor in video_data["video_processor"]:
                processor = ProcessorsFactory().build(processor["name"], processor["config"])
                processor.process([video_dir], batch_size=batch_size)
        else:
            self.logger.info(f"Processing for source {source} is not supported. In Video data {video_data}.")

    def process_video_by_id(self, source: str, video_id: str, batch_size: int = -1) -> None:
        """Process a single video identified by its source and video ID.

        Retrieves the video data from the pipeline's state and applies the configured processors.
        If the specified source or video ID does not exist, logs an informational message and exits.

        Args:
            source: The source platform of the video (e.g., "youtube").
            video_id: The unique identifier of the video within the source platform.
            batch_size: Number of batches to use during processing..

        Returns:
            None
        """
        if source not in self.video_item_dict:
            self.logger.info(f"Source '{source}' not found in pipeline. Skipping processing.")
            return
        if video_id not in self.video_item_dict[source]:
            self.logger.info(f"Video ID '{video_id}' not found in source '{source}'. Skipping processing.")
            return
        video_data = self.video_item_dict[source][video_id]
        self.process_video(video_data=video_data, batch_size=batch_size)

    def process_video_by_id_overwrite_processor(
        self, source: str, video_id: str, processor_name: str, processor_config: Dict[str, Any], batch_size: int = -1
    ) -> None:
        """Process a single video identified by its source and video ID using processor.

        Retrieves the video data from the pipeline's state and applies the configured processors.
        If the specified source or video ID does not exist, logs an informational message and exits.
        Uses the specified processor.

        Args:
            source: The source platform of the video (e.g., "youtube").
            video_id: The unique identifier of the video within the source platform.
            processor_name: The name of the porceossor.
            processor_config: The configuration for the processor.
            batch_size: Number of batches to use during processing..

        Returns:
            None
        """
        if source not in self.video_item_dict:
            self.logger.info(f"Source '{source}' not found in pipeline. Skipping processing.")
            return
        if video_id not in self.video_item_dict[source]:
            self.logger.info(f"Video ID '{video_id}' not found in source '{source}'. Skipping processing.")
            return
        video_data = copy.deepcopy(self.video_item_dict[source][video_id])
        video_data["video_processor"] = [
            {
                "name": processor_name,
                "config": processor_config,
            }
        ]
        self.process_video(video_data=video_data, batch_size=batch_size)

    def process(self, batch_size: int = -1):
        """Processes all video entries in the Pipeline that have not been processed yet."""
        for source in self.video_item_dict.keys():
            for video_id in tqdm(self.video_item_dict[source].keys(), desc="Processing video"):
                video_dir = (self.base_dir / source) / video_id
                _, video_metadata = load_metadata(video_dir=video_dir)
                if not video_metadata["processed"]:
                    self.process_video(video_data=self.video_item_dict[source][video_id], batch_size=batch_size)

    def process_overwrite_processor(self, processor_name: str, processor_config: Dict[str, Any], batch_size: int = -1):
        """Processes all video entries in the Pipeline with the specified processor.

        Please note that this is not the intended way to process the pipeline entries
        since the Pipeline object does not keep track, which processor was used.

        Args:
            processor_name: The name of the porceossor.
            processor_config: The configuration for the processor.
            batch_size: Gets propgaed to the processor.
        """
        for source in self.video_item_dict.keys():
            for video_id in tqdm(self.video_item_dict[source].keys(), desc="Processing video"):
                self.process_video_by_id_overwrite_processor(
                    source=source,
                    video_id=video_id,
                    processor_name=processor_name,
                    processor_config=processor_config,
                    batch_size=batch_size,
                )

    def remove_ref_frame_variations(self):
        """Removes all generated reference frame variations and updates the metadata json."""
        for source in self.video_item_dict.keys():
            for video_id in tqdm(self.video_item_dict[source].keys(), desc="Removing ref frame varaitions"):
                video_path = self.base_dir / source / video_id
                clean_ref_frame_variations(video_dir_list=[str(video_path)])

"""Unittest for InputFile."""

import dataclasses
import json
import os
import tempfile
import unittest
from typing import Any, List

from painvidpro.pipeline.input_file_format import VideoItem, VideoProcessor


class TestVideoClass(unittest.TestCase):
    def setUp(self):
        self.test_instance_1 = VideoItem(
            source="youtube",
            url="https://www.youtube.com/watch?v=OWk-AMkTOy4&pp=ygUJcGFpbnRsYW5l",
            is_playlist=False,
            art_media=["pencil"],
            channel_ids=[],
            video_processor=[VideoProcessor(name="KeyframeExtractor", config={})],
        )
        self.test_instance_2 = VideoItem(
            source="youtube",
            url="https://www.youtube.com/watch?v=hkJAAqwJzn4&list=PLxXQD4NIqhjQ4cSfPuvrDqAQHLlK19CTy",
            is_playlist=True,
            art_media=["pencil"],
            channel_ids=[],
            video_processor=[VideoProcessor(name="KeyframeExtractor", config={})],
        )
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "test_video_class.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_to_json(self):
        with open(self.file_path, "w") as file:
            json.dump(dataclasses.asdict(self.test_instance_1), file)
            file.write("\n")
            json.dump(dataclasses.asdict(self.test_instance_2), file)
            file.write("\n")

        video_item_list: List[Any] = []
        with open(self.file_path, "r") as file:
            for line in file:
                video_item_list.append(json.loads(line))

        loaded_instance_1 = VideoItem(**video_item_list[0])
        loaded_instance_2 = VideoItem(**video_item_list[1])

        self.assertEqual(dataclasses.asdict(self.test_instance_1), dataclasses.asdict(loaded_instance_1))
        self.assertEqual(dataclasses.asdict(self.test_instance_2), dataclasses.asdict(loaded_instance_2))


if __name__ == "__main__":
    unittest.main()

"""Unittest class."""

import os
import unittest
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from painvidpro.occlusion_removing.factory import OcclusionRemovingFactory
from painvidpro.occlusion_removing.frame_merging import OcclusionRemovingFrameMerging


class TestOcclusionRemovingFrameMerging(unittest.TestCase):
    def setUp(self):
        self.factory_merger = OcclusionRemovingFactory().build("OcclusionRemovingFrameMerging", {})
        self.merger = OcclusionRemovingFrameMerging()
        self.fill_color = (0, 0, 0)
        self.num_frames = 3
        self.height, self.width = 100, 100

        # Create temporary directories for input and output
        self.temp_dir = TemporaryDirectory()
        self.output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Create dummy frames and masks
        self.frame_list = []
        self.mask_list = []
        self.frame_paths = []
        self.mask_paths = []
        for i in range(self.num_frames):
            frame = np.full((self.height, self.width, 3), i * 50, dtype=np.uint8)
            mask = np.full((self.height, self.width), 255 if i % 2 == 0 else 0, dtype=np.uint8)
            self.frame_list.append(frame)
            self.mask_list.append(mask)
            frame_path = os.path.join(self.temp_dir.name, f"frame_{i}.png")
            mask_path = os.path.join(self.temp_dir.name, f"mask_{i}.png")
            cv2.imwrite(frame_path, frame)
            cv2.imwrite(mask_path, mask)
            self.frame_paths.append(frame_path)
            self.mask_paths.append(mask_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_create_forward_backward_frames_and_masks_on_disk(self):
        forward_frame_paths, forward_mask_paths, backward_frame_paths, backward_mask_paths = (
            self.merger.create_forward_backward_frames_and_masks_on_disk(
                self.frame_paths, self.mask_paths, self.output_dir, self.fill_color
            )
        )

        # Check if the output files are created
        self.assertEqual(len(forward_frame_paths), self.num_frames)
        self.assertEqual(len(forward_mask_paths), self.num_frames)
        self.assertEqual(len(backward_frame_paths), self.num_frames)
        self.assertEqual(len(backward_mask_paths), self.num_frames)

        for path in forward_frame_paths + forward_mask_paths + backward_frame_paths + backward_mask_paths:
            self.assertTrue(os.path.exists(path))

    def test_create_merged_frame_list_on_disk(self):
        forward_frame_paths, forward_mask_paths, backward_frame_paths, backward_mask_paths = (
            self.merger.create_forward_backward_frames_and_masks_on_disk(
                self.frame_paths, self.mask_paths, self.output_dir, self.fill_color
            )
        )

        merged_frame_paths = self.merger.create_merged_frame_list_on_disk(
            self.frame_paths,
            self.mask_paths,
            forward_frame_paths,
            forward_mask_paths,
            backward_frame_paths,
            backward_mask_paths,
            self.output_dir,
        )

        # Check if the output files are created
        self.assertEqual(len(merged_frame_paths), self.num_frames)

        for path in merged_frame_paths:
            self.assertTrue(os.path.exists(path))

    def test_remove_occlusions(self):
        merged_frames = self.factory_merger.remove_occlusions(frame_list=self.frame_list, mask_list=self.mask_list)
        merged_frames_path = self.factory_merger.remove_occlusions_on_disk(
            self.frame_paths, self.mask_paths, self.output_dir
        )

        for frame, frame_path in zip(merged_frames, merged_frames_path):
            frame_from_disk = cv2.imread(frame_path)
            self.assertTrue(np.array_equal(frame, frame_from_disk))


if __name__ == "__main__":
    unittest.main()

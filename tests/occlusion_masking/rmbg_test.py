"""RMBG Test."""

import unittest
from unittest.mock import MagicMock, call

import numpy as np
import torch

from painvidpro.occlusion_masking.factory import OcclusionMaskingFactory


class TestComputeMaskList(unittest.TestCase):
    def setUp(self):
        self.obj = OcclusionMaskingFactory().build("OcclusionMaskingRMBG", {})
        self.obj.model = MagicMock()
        self.obj.params = {"device": "cpu", "batch_size": 1}
        self.mock_output = (torch.zeros(1, 1, 1024, 1024),)

    def test_valid_list_input(self):
        frame_list = [
            np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8),
            np.random.randint(0, 256, (150, 250, 3), dtype=np.uint8),
        ]
        self.obj.model.return_value = self.mock_output
        masks = self.obj.compute_mask_list(frame_list)
        self.assertEqual(len(masks), 2)
        self.assertEqual(masks[0].shape, (100, 200))
        self.assertEqual(masks[1].shape, (150, 250))
        self.assertTrue(all(m.dtype == bool for m in masks))

    def test_valid_4d_array_input(self):
        frame_array = np.random.randint(0, 256, (3, 100, 200, 3), dtype=np.uint8)
        self.obj.model.return_value = self.mock_output
        masks = self.obj.compute_mask_list(frame_array)
        self.assertEqual(len(masks), 3)
        for mask in masks:
            self.assertEqual(mask.shape, (100, 200))

    def test_valid_3d_array_input(self):
        frame_array = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)
        self.obj.model.return_value = self.mock_output
        masks = self.obj.compute_mask_list(frame_array)
        self.assertEqual(len(masks), 1)
        self.assertEqual(masks[0].shape, (100, 200))

    def test_invalid_2d_array_input(self):
        frame_array = np.random.randint(0, 256, (100, 200), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.obj.compute_mask_list(frame_array)

    def test_model_not_initialized(self):
        self.obj.model = None
        frame_list = [np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)]
        with self.assertRaises(ValueError):
            self.obj.compute_mask_list(frame_list)

    def test_batching(self):
        def batched_out(input):
            b = input.shape[0]
            return (torch.zeros(b, 1, 1024, 1024),)

        self.obj.params["batch_size"] = 2
        frame_list = [np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8) for _ in range(3)]
        self.obj.model.side_effect = batched_out
        masks = self.obj.compute_mask_list(frame_list)
        self.assertEqual(self.obj.model.call_count, 2)
        self.assertEqual(len(masks), 3)

    def test_offload_model_true(self):
        self.obj.model.to = MagicMock(return_value=self.obj.model)
        self.obj.model.return_value = self.mock_output
        self.obj.params["device"] = "cuda"
        frame_list = [np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)]
        self.obj.compute_mask_list(frame_list, offload_model=True)
        self.obj.model.to.assert_has_calls([call("cuda"), call("cpu")])

    def test_offload_model_false(self):
        self.obj.model.to = MagicMock(return_value=self.obj.model)
        self.obj.model.return_value = self.mock_output
        self.obj.params["device"] = "cuda"
        frame_list = [np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8)]
        self.obj.compute_mask_list(frame_list, offload_model=False)
        self.obj.model.to.assert_called_once_with("cuda")


if __name__ == "__main__":
    unittest.main()

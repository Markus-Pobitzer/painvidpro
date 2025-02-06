import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from painvidpro.sequence_detection.grounding_dino import SequenceDetectionGroundingDino


class TestSequenceDetectionGroundingDino(unittest.TestCase):
    def setUp(self):
        self.detector = SequenceDetectionGroundingDino()
        # Mock model and processor to avoid real model loading
        self.mock_model = MagicMock()
        self.mock_processor = MagicMock()
        self.mock_post_process = MagicMock(return_value=[{"labels": ["hand"], "scores": torch.tensor([0.5])}])

    def test_initialization(self):
        self.assertEqual(self.detector.params["model_id"], "IDEA-Research/grounding-dino-tiny")
        self.assertIsNone(self.detector.model)
        self.assertIsNone(self.detector.processor)

    def test_set_parameters(self):
        with (
            patch("transformers.AutoProcessor.from_pretrained", return_value=self.mock_processor),
            patch("transformers.AutoModelForZeroShotObjectDetection.from_pretrained", return_value=self.mock_model),
        ):
            success, msg = self.detector.set_parameters({"model_id": "mock-model"})
            self.assertTrue(success)
            self.assertEqual(msg, "")
            self.assertEqual(self.detector.params["model_id"], "mock-model")
            self.assertEqual(self.detector.model, self.mock_model)
            self.assertEqual(self.detector.processor, self.mock_processor)

    def test_object_detection(self):
        self.detector.processor = self.mock_processor
        self.detector.model = self.mock_model
        self.mock_processor.post_process_grounded_object_detection = self.mock_post_process

        images = [np.zeros((100, 100, 3), dtype=np.uint8)]
        labels, scores = self.detector._object_detection(images)
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0], ["hand"])
        self.assertEqual(scores[0].tolist(), [0.5])

    def test_detect_first_occurrence_found(self):
        self.detector._object_detection = MagicMock(return_value=([["hand"], []], [[0.5], []]))
        frame_idxs = [0, 1]
        frames = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))]
        result = self.detector._detect_first_occurence(frame_idxs, frames)
        self.assertEqual(result, 0)

    def test_detect_first_occurrence_not_found(self):
        self.detector._object_detection = MagicMock(return_value=([[], []], [[], []]))
        frame_idxs = [0, 1]
        frames = [np.zeros((100, 100, 3)), np.zeros((100, 100, 3))]
        result = self.detector._detect_first_occurence(frame_idxs, frames)
        self.assertEqual(result, -1)

    def test_detect_sequences_no_detections(self):
        self.detector._detect_first_occurence = MagicMock(return_value=-1)
        self.detector.processor = self.mock_processor
        self.detector.model = self.mock_model
        frames = [np.zeros((100, 100, 3))] * 10
        sequences = self.detector.detect_sequences(frames, True)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0].start_idx, 0)
        self.assertEqual(sequences[0].end_idx, 10)

    def test_detect_sequences_all_detections(self):
        self.detector._detect_first_occurence = MagicMock(side_effect=[0, 9])
        self.detector.processor = self.mock_processor
        self.detector.model = self.mock_model
        frames = [np.zeros((100, 100, 3))] * 10
        sequences = self.detector.detect_sequences(frames, False)
        self.assertEqual(sequences[0].start_idx, 0)
        self.assertEqual(sequences[0].end_idx, 9)

    def test_detect_sequences_edge_cases(self):
        # Empty frame list
        with self.assertRaises(ValueError):
            self.detector.detect_sequences([], True)

        # Single frame
        self.detector._detect_first_occurence = MagicMock(return_value=0)
        self.detector.processor = self.mock_processor
        self.detector.model = self.mock_model
        sequences = self.detector.detect_sequences([np.zeros((100, 100, 3))], True)
        self.assertEqual(sequences[0].start_idx, 0)
        self.assertEqual(sequences[0].end_idx, 0)


if __name__ == "__main__":
    unittest.main()

"""Base class for the Occlusion Removing."""

import os
from typing import Any, Dict, List, Tuple

import numpy as np


class OcclusionRemovingBase:
    def __init__(self):
        """Base class to remove occlusions."""
        self.params: Dict[str, Any] = {}

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        self.params.update(params)
        return True, ""

    def set_default_parameters(self):
        raise NotImplementedError("This method should be implemented by the child class.")

    def remove_files_from_disk(self, file_list: List[str]):
        """
        Removes files from disk given lists of file paths.

        Args:
            file_lists: Lists of file paths to be removed.
        """
        for file_path in file_list:
            if os.path.exists(file_path):
                os.remove(file_path)

    def remove_occlusions(self, frame_list: List[np.ndarray], mask_list: List[np.ndarray]) -> List[np.ndarray]:
        """
        Removes occlusions indicated by the masks.

        Args:
            frame_list: List of frames in cv2 image format.
            mask_list: List of masks in cv2 format.

        Returns:
            List of frames where the parts of the frame indicated
            by the masks (occlusions) have been removed.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

    def remove_occlusions_on_disk(
        self, frame_path_list: List[str], mask_path_list: List[str], output_dir: str
    ) -> List[str]:
        """
        Removes occlusions indicated by the masks.

        Images are saved to disk and only the paths get returned.

        Args:
            frame_path_list: List of paths to frames representing the video.
            mask_path_list: List of paths to masks corresponding to each frame.
            output_dir: Directory where the output merged frames will be stored.

        Returns:
            List of frame paths where the parts of the frame indicated
            by the masks (occlusions) have been removed.
        """
        raise NotImplementedError("This method should be implemented by the child class.")

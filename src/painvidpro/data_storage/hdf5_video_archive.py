import json
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
from h5py import Dataset, File
from PIL import Image


class DynamicVideoArchive:
    """
    A flexible HDF5 wrapper that allows:
    1. Setting global metadata before knowing video dimensions.
    2. Appending frames (growing the file).
    3. Updating specific frames (random access).
    """

    def __init__(self, filename: str, mode: str = "a"):
        self.filename: str = filename
        self.mode: str = mode
        self._file: Optional[File] = None

    @property
    def file(self) -> File:
        if self._file is None:
            raise RuntimeError(
                f"To modify entries of the {self.__class__} class make sure to call it inside a `with`."
            )
        return self._file

    @property
    def frames_dset(self) -> Dataset:
        if "frames" not in self.file:
            raise RuntimeError("Storage not initialized. Add a frame first.")
        return self.file["frames"]  # type: ignore

    @property
    def meta_dset(self) -> Dataset:
        if "frame_metadata" not in self.file:
            raise RuntimeError("Storage not initialized. Add a frame first.")
        return self.file["frame_metadata"]  # type: ignore

    def __enter__(self):
        self._file = File(self.filename, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __len__(self):
        """
        Allows len(archive) to get the frame count.
        Returns 0 if storage hasn't been initialized yet.
        """
        if "frames" in self.file:
            return self.frames_dset.shape[0]
        return 0

    # --- 1. GLOBAL METADATA ---
    def set_global_metadata(self, key: str, value: Any):
        self.file.attrs[key] = value

    def get_global_metadata(self):
        return dict(self.file.attrs)

    # --- 2. STORAGE SETUP ---
    def _ensure_storage(self, width, height):
        """Internal helper to create datasets if they don't exist."""
        if "frames" in self.file:
            return

        self.file.create_dataset(
            "frames",
            shape=(0, height, width, 3),
            maxshape=(None, height, width, 3),
            dtype="uint8",
            chunks=(1, height, width, 3),
            compression="gzip",
        )

        dt_str = h5py.string_dtype(encoding="utf-8")
        self.file.create_dataset("frame_metadata", shape=(0,), maxshape=(None,), dtype=dt_str)

    # --- 3. WRITE OPERATIONS ---

    def add_frame(self, image: Image.Image, metadata_dict: Dict[str, Any]):
        """Appends a new frame to the end of the video."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        self._ensure_storage(image.width, image.height)

        # Resize to fit one more frame
        new_index = self.frames_dset.shape[0]
        self.frames_dset.resize(new_index + 1, axis=0)
        self.meta_dset.resize(new_index + 1, axis=0)

        # Write data
        self.frames_dset[new_index] = np.asarray(image)
        self.meta_dset[new_index] = json.dumps(metadata_dict)

    def update_frame(self, index: int, image: Image.Image, metadata_dict: Dict[str, Any]):
        """
        Overwrites an EXISTING frame at the specified index.
        """
        total_frames = self.frames_dset.shape[0]
        if index < 0 or index >= total_frames:
            raise IndexError(f"Index {index} is out of bounds for video with {total_frames} frames.")

        if image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Write Data (No resizing needed)
        self.frames_dset[index] = np.asarray(image)
        self.meta_dset[index] = json.dumps(metadata_dict)

    # --- 4. READ OPERATIONS ---

    def get_frame(self, index: int) -> Tuple[Image.Image, Dict[str, Any]]:
        """Returns (PIL.Image, dict)"""
        if "frames" not in self.file:
            raise RuntimeError("No frames stored yet.")

        # Read from disk
        arr = self.frames_dset[index]
        json_str = self.meta_dset[index]

        # Convert
        image = Image.fromarray(arr)
        if isinstance(json_str, bytes):
            json_str = json_str.decode("utf-8")

        return image, json.loads(json_str)

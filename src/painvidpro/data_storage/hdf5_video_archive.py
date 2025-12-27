import io
import json
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np
from h5py import Dataset, File
from PIL import Image


class DynamicVideoArchive:
    """
    A high-level interface for managing video data and metadata in HDF5 format.

    This class provides a context-managed wrapper around HDF5 files, specifically
    optimized for storing video frames as JPEG-compressed binary blobs. This
    approach significantly reduces disk space compared to raw pixel storage
    while maintaining fast random access to individual frames.

    Key Features:
    - **Deferred Initialization**: Global metadata can be set before video
      dimensions are known.
    - **Dynamic Growth**: Supports appending frames to an existing archive.
    - **Random Access**: Specific frames can be retrieved or updated by index.
    - **JPEG Compression**: Built-in lossy compression for efficient storage.

    Example:
        >>> archive = DynamicVideoArchive("recording.h5", quality=90)
        >>> with archive:
        ...     archive.set_global_metadata("session_id", 42)
        ...     for img in my_frames:
        ...         archive.add_frame(img, {"timestamp": 1625097600})

    Attributes:
        filename (str): Path to the HDF5 file.
        mode (str): File access mode (e.g., 'a' for append/read/write, 'w' for write).
        quality (int): JPEG compression quality (1-100).
    """

    def __init__(self, filename: str, mode: str = "a", quality: int = 85):
        """
        Initializes the DynamicVideoArchive instance.

        Args:
            filename (str): The path to the .h5 file to create or open.
            mode (str, optional): The mode in which the file is opened.
                'a' (default) reads/writes and creates the file if it doesn't exist.
                'w' creates a new file (truncating existing).
                'r' for read-only.
            quality (int, optional): The quality of JPEG compression, where 100
                is best quality/least compression. Defaults to 85.

        Raises:
            ValueError: If the quality is not within the range [1, 100].
        """
        self.filename: str = str(filename)
        self.mode: str = mode
        self.quality = quality  # JPEG compression quality
        self._file: Optional[File] = None

    @property
    def file(self) -> File:
        """
        The underlying HDF5 file object.

        Returns:
            h5py.File: The open HDF5 file handle.

        Raises:
            RuntimeError: If accessed outside of a 'with' context block.
        """
        if self._file is None:
            raise RuntimeError(f"To use the {__name__} class make sure to call it inside a `with`.")
        return self._file

    @property
    def frames_dset(self) -> Dataset:
        """
        The HDF5 dataset containing the compressed video frames.

        Returns:
            h5py.Dataset: Dataset of variable-length uint8 arrays.

        Raises:
            RuntimeError: If the storage hasn't been initialized by adding a frame.
        """
        if "frames" not in self.file:
            raise RuntimeError("Storage not initialized. Add a frame first.")
        return self.file["frames"]  # type: ignore

    @property
    def meta_dset(self) -> Dataset:
        """
        The HDF5 dataset containing per-frame metadata (JSON strings).

        Returns:
            h5py.Dataset: Dataset of variable-length UTF-8 strings.
        """
        if "frame_metadata" not in self.file:
            raise RuntimeError("Storage not initialized. Add a frame first.")
        return self.file["frame_metadata"]  # type: ignore

    @property
    def reference_frames_dset(self) -> Dataset:
        """
        The HDF5 dataset containing compressed reference/calibration frames.

        Returns:
            h5py.Dataset: Dataset of variable-length uint8 arrays.
        """
        if "reference_frames" not in self.file:
            raise RuntimeError("Storage not initialized. Add a frame first.")
        return self.file["reference_frames"]  # type: ignore

    @property
    def reference_meta_dset(self) -> Dataset:
        """
        The HDF5 dataset containing metadata for reference frames.

        Returns:
            h5py.Dataset: Dataset of variable-length UTF-8 strings.
        """
        if "reference_frame_metadata" not in self.file:
            raise RuntimeError("Storage not initialized. Add a frame first.")
        return self.file["reference_frame_metadata"]  # type: ignore

    def __enter__(self):
        """
        Opens the HDF5 file and enters the runtime context.

        Returns:
            DynamicVideoArchive: The instance with an active file handle.
        """
        self._file = File(self.filename, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the HDF5 file and exits the runtime context.

        Args:
            exc_type: Exception type if an error occurred.
            exc_val: Exception value.
            exc_tb: Exception traceback.
        """
        if self._file is not None:
            self._file.close()
            self._file = None

    def __len__(self):
        """
        Returns the total number of frames currently stored in the archive.

        Returns:
            int: The number of frames. Returns 0 if the dataset is not yet created.
        """
        if "frames" in self.file:
            return self.frames_dset.shape[0]
        return 0

    def __getitem__(self, index: int) -> Image.Image:
        """Shorthand for get_frame(index)."""
        return self.get_frame(index)

    def len_reference_frames(self):
        """
        Returns the total number of reference frames currently stored in the archive.

        Returns:
            int: The number of referenceframes. Returns 0 if the dataset is not yet created.
        """
        if "reference_frames" in self.file:
            return self.reference_frames_dset.shape[0]
        return 0

    # --- 1. GLOBAL METADATA ---
    def update_global_metadata(self, metadata: Dict[str, Any]):
        """
        Updates the HDF5 file root attributes with multiple key-value pairs.

        Args:
            metadata (Dict[str, Any]): A dictionary of metadata to store
                at the file level.
        """
        self.file.attrs.update(metadata)

    def set_global_metadata(self, key: str, value: Any):
        """
        Sets a single global metadata attribute.

        Args:
            key (str): The name of the metadata field.
            value (Any): The value to store (must be HDF5-compatible).
        """
        self.file.attrs[key] = value

    def get_global_metadata(self):
        """
        Retrieves all global metadata stored at the file root.

        Returns:
            Dict[str, Any]: A dictionary of all file attributes.
        """
        return dict(self.file.attrs)

    def _prepare_image(self, image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Internal helper to transform an image into a JPEG-compressed binary blob.

        This method handles format conversion (ensuring RGB) and utilizes
        in-memory byte buffers to avoid disk I/O during compression.

        Args:
            image (Union[PIL.Image.Image, np.ndarray]): The input frame.
                Can be a PIL Image object or a NumPy array (H, W, C).

        Returns:
            np.ndarray: A 1D NumPy array of dtype 'uint8' containing
                the raw JPEG byte stream.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Compress to JPEG in memory
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        jpeg_data = np.frombuffer(buffer.getvalue(), dtype="uint8")
        return jpeg_data

    # --- 2. STORAGE SETUP ---
    def _ensure_storage(
        self,
    ):
        """
        Initializes the HDF5 internal structure if it has not been created yet.

        This method sets up four primary datasets using variable-length types:
        1. **frames**: For JPEG-compressed video data.
        2. **frame_metadata**: For JSON-encoded strings linked to each frame.
        3. **reference_frames**: For compressed reference frames.
        4. **reference_frame_metadata**: For metadata linked to reference frames.

        The datasets are configured with `maxshape=(None,)` to allow for
        infinite appending (unlimited growth along the first axis).
        Chunking is set to 1 to optimize for sequential and random access
        of individual compressed blobs.

        Note:
            This is an idempotent internal method called automatically by
            `add_frame` and `add_reference_frame`.
        """
        if "frames" in self.file:
            return

        # Use a variable-length uint8 type to store binary blobs (JPEGs)
        vlen_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        # Dtype for metadata
        dt_str = h5py.string_dtype(encoding="utf-8")

        # Frame Dataset
        self.file.create_dataset(
            "frames",
            shape=(0,),
            maxshape=(None,),
            dtype=vlen_dtype,
            chunks=(1,),  # Chunking per frame is efficient for variable length
        )
        self.file.create_dataset("frame_metadata", shape=(0,), maxshape=(None,), dtype=dt_str)

        # Reference Frame Dataset
        self.file.create_dataset(
            "reference_frames",
            shape=(0,),
            maxshape=(None,),
            dtype=vlen_dtype,
            chunks=(1,),  # Chunking per frame is efficient for variable length
        )
        self.file.create_dataset("reference_frame_metadata", shape=(0,), maxshape=(None,), dtype=dt_str)

    # --- 3. WRITE OPERATIONS ---

    def add_frame(self, image: Union[Image.Image, np.ndarray], metadata_dict: Dict[str, Any]):
        """
        Appends a new video frame and its associated metadata to the archive.

        This method automatically handles storage initialization on the first call,
        compresses the image to JPEG format, and expands the dataset to
        accommodate the new entry.

        Args:
            image (Union[PIL.Image.Image, np.ndarray]): The frame to add.
            metadata_dict (Dict[str, Any]): Metadata specific to this frame
                (e.g., timestamps, sensor readings), which will be stored as JSON.
        """
        self._ensure_storage()

        # Resize to fit one more frame
        new_index = self.frames_dset.shape[0]
        self.frames_dset.resize(new_index + 1, axis=0)
        self.meta_dset.resize(new_index + 1, axis=0)

        # Write data
        self.frames_dset[new_index] = self._prepare_image(image=image)
        self.meta_dset[new_index] = json.dumps(metadata_dict)
        # Consider flushing the data
        # self.file.flush()

    def update_frame(self, index: int, image: Union[Image.Image, np.ndarray], metadata_dict: Dict[str, Any]):
        """
        Overwrites an existing video frame and its metadata at a specific index.

        Args:
            index (int): The zero-based index of the frame to overwrite.
            image (Union[PIL.Image.Image, np.ndarray]): The new frame data.
            metadata_dict (Dict[str, Any]): The new metadata dictionary.

        Raises:
            IndexError: If the index is outside the range of currently stored frames.
        """
        total_frames = self.frames_dset.shape[0]
        if index < 0 or index >= total_frames:
            raise IndexError(f"Index {index} is out of bounds for video with {total_frames} frames.")

        # 2. Write Data (No resizing needed)
        self.frames_dset[index] = self._prepare_image(image=image)
        self.meta_dset[index] = json.dumps(metadata_dict)

    def add_reference_frame(self, image: Union[Image.Image, np.ndarray], metadata_dict: Dict[str, Any]):
        """
        Appends a new reference frame to the archive.

        Reference frames are stored in a separate dataset from the main video
        stream, allowing for easy separation.

        Args:
            image (Union[PIL.Image.Image, np.ndarray]): The reference frame to add.
            metadata_dict (Dict[str, Any]): Metadata specific to this reference frame.
        """
        self._ensure_storage()

        # Resize to fit one more frame
        new_index = self.reference_frames_dset.shape[0]
        self.reference_frames_dset.resize(new_index + 1, axis=0)
        self.reference_meta_dset.resize(new_index + 1, axis=0)

        # Write data
        self.reference_frames_dset[new_index] = self._prepare_image(image=image)
        self.reference_meta_dset[new_index] = json.dumps(metadata_dict)

    def update_reference_frame(self, index: int, image: Union[Image.Image, np.ndarray], metadata_dict: Dict[str, Any]):
        """
        Overwrites an existing reference frame at a specific index.

        Args:
            index (int): The zero-based index of the reference frame to overwrite.
            image (Union[PIL.Image.Image, np.ndarray]): The new reference data.
            metadata_dict (Dict[str, Any]): The new metadata dictionary.

        Raises:
            IndexError: If the index is out of bounds for the reference dataset.
        """
        total_frames = self.frames_dset.shape[0]
        if index < 0 or index >= total_frames:
            raise IndexError(f"Index {index} is out of bounds for video with {total_frames} frames.")

        # Write Data
        self.reference_frames_dset[index] = self._prepare_image(image=image)
        self.reference_meta_dset[index] = json.dumps(metadata_dict)

    def remove_reference_frame(self, index: int):
        """
        Removes a reference frame at the given index and shifts subsequent frames.

        Args:
            index (int): The zero-based index of the reference frame to overwrite.

        Raises:
            IndexError: If the index is out of bounds for the reference dataset.
        """
        self._ensure_storage()

        current_shape = self.reference_frames_dset.shape[0]

        if not 0 <= index < current_shape:
            raise IndexError(f"Index {index} out of range for dataset with size {current_shape}")

        # Shift data if the element is not the last one
        if index < current_shape - 1:
            # Move all frames after 'index' up by one position
            self.reference_frames_dset[index:-1] = self.reference_frames_dset[index + 1 :]
            self.reference_meta_dset[index:-1] = self.reference_meta_dset[index + 1 :]

        # Shrink the dataset size by 1
        self.reference_frames_dset.resize(current_shape - 1, axis=0)
        self.reference_meta_dset.resize(current_shape - 1, axis=0)

    # --- 4. READ OPERATIONS ---

    def get_frame(self, index: int) -> Image.Image:
        """
        Retrieves and decompresses a specific video frame from the archive.

        This method reads the compressed binary blob from the HDF5 file and
        reconstructs it into a PIL Image object using an in-memory buffer.

        Args:
            index (int): The zero-based index of the frame to retrieve.

        Returns:
            PIL.Image.Image: The decompressed RGB image.

        Raises:
            IndexError: If the index is out of bounds.
        """
        binary_data = self.frames_dset[index]
        return Image.open(io.BytesIO(binary_data))

    def get_frame_metadata(self, index: int) -> Dict[str, Any]:
        """
        Retrieves the metadata associated with a specific video frame.

        Args:
            index (int): The zero-based index of the metadata to retrieve.

        Returns:
            Dict[str, Any]: The metadata dictionary reconstructed from JSON.
        """
        # Read from disk
        json_str = self.meta_dset[index]

        if isinstance(json_str, bytes):
            json_str = json_str.decode("utf-8")

        return json.loads(json_str)

    def get_reference_frame(self, index: int) -> Image.Image:
        """
        Retrieves and decompresses a specific reference frame.

        Args:
            index (int): The zero-based index of the reference frame.

        Returns:
            PIL.Image.Image: The decompressed reference image.
        """
        binary_data = self.reference_frames_dset[index]
        return Image.open(io.BytesIO(binary_data))

    def get_reference_frame_metadata(self, index: int) -> Dict[str, Any]:
        """
        Retrieves the metadata associated with a specific reference frame.

        Args:
            index (int): The zero-based index of the reference metadata.

        Returns:
            Dict[str, Any]: The reconstructed metadata dictionary.
        """
        # Read from disk
        json_str = self.reference_meta_dset[index]

        if isinstance(json_str, bytes):
            json_str = json_str.decode("utf-8")

        return json.loads(json_str)

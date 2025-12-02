import h5py
import numpy as np
import json
from PIL import Image

class DynamicVideoArchive:
    """
    A flexible HDF5 wrapper that allows:
    1. Setting global metadata before knowing video dimensions.
    2. Appending frames (growing the file).
    3. Updating specific frames (random access).
    """

    def __init__(self, filename, mode='a'):
        self.filename = filename
        self.mode = mode
        self.file = None
        self.frames_dset = None
        self.meta_dset = None

    def __enter__(self):
        self.file = h5py.File(self.filename, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def __len__(self):
        """
        Allows len(archive) to get the frame count.
        Returns 0 if storage hasn't been initialized yet.
        """
        if self.file and "frames" in self.file:
            return self.file['frames'].shape[0]
        return 0

    # --- 1. GLOBAL METADATA ---
    def set_global_metadata(self, key, value):
        self.file.attrs[key] = value

    def get_global_metadata(self):
        return dict(self.file.attrs)

    # --- 2. STORAGE SETUP ---
    def _ensure_storage(self, width, height):
        """Internal helper to create datasets if they don't exist."""
        if "frames" in self.file:
            return

        self.frames_dset = self.file.create_dataset(
            "frames",
            shape=(0, height, width, 3),
            maxshape=(None, height, width, 3),
            dtype='uint8',
            chunks=(1, height, width, 3),
            compression="gzip"
        )

        dt_str = h5py.string_dtype(encoding='utf-8')
        self.meta_dset = self.file.create_dataset(
            "frame_metadata",
            shape=(0,),
            maxshape=(None,),
            dtype=dt_str
        )

    # --- 3. WRITE OPERATIONS ---

    def add_frame(self, pil_image, metadata_dict):
        """Appends a NEW frame to the end of the video."""
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
            
        self._ensure_storage(pil_image.width, pil_image.height)
        
        # Reload references
        self.frames_dset = self.file['frames']
        self.meta_dset = self.file['frame_metadata']

        # Resize to fit one more frame
        new_index = self.frames_dset.shape[0]
        self.frames_dset.resize(new_index + 1, axis=0)
        self.meta_dset.resize(new_index + 1, axis=0)

        # Write data
        self.frames_dset[new_index] = np.asarray(pil_image)
        self.meta_dset[new_index] = json.dumps(metadata_dict)

    def update_frame(self, index, pil_image, metadata_dict):
        """
        Overwrites an EXISTING frame at the specified index.
        """
        # 1. Safety Checks
        if "frames" not in self.file:
            raise RuntimeError("Storage not initialized. Add a frame first.")
        
        total_frames = self.file['frames'].shape[0]
        if index < 0 or index >= total_frames:
            raise IndexError(f"Index {index} is out of bounds for video with {total_frames} frames.")

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        # 2. Write Data (No resizing needed)
        self.file['frames'][index] = np.asarray(pil_image)
        self.file['frame_metadata'][index] = json.dumps(metadata_dict)

    # --- 4. READ OPERATIONS ---

    def get_frame(self, index):
        """Returns (PIL.Image, dict)"""
        if "frames" not in self.file:
            raise RuntimeError("No frames stored yet.")
            
        # Read from disk
        arr = self.file['frames'][index]
        json_str = self.file['frame_metadata'][index]
        
        # Convert
        image = Image.fromarray(arr)
        if isinstance(json_str, bytes):
            json_str = json_str.decode('utf-8')
            
        return image, json.loads(json_str)
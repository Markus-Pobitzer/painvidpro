from pathlib import Path
from typing import List

from painvidpro.data_storage.hdf5_video_archive import DynamicVideoArchive


def clean_ref_frame_variations(video_dir_list: List[str], frame_data_file="frame_data.h5"):
    """
    Removes the 'reference_frame_variations' entry from the metadata and deletes the corresponding variations directory for each video directory in the provided list.

    Args:
        video_dir_list (List[str]): List of paths to video directories.
        frame_data_file (str): Name of the DynamicVideoArchive.

    Side Effects:
        - Modifies the metadata by removing the 'reference_frame_variations' key if present.
        - Deletes the variations directory within each video directory.
    """
    for video_dir in video_dir_list:
        frame_data_path = Path(video_dir) / frame_data_file
        with DynamicVideoArchive(filename=str(frame_data_path)) as archive:
            num_reference_frames = archive.len_reference_frames()
            if num_reference_frames > 1:
                # Remove every reference frame except the first one
                for idx in range(num_reference_frames - 1, 0, -1):
                    archive.remove_reference_frame(idx)

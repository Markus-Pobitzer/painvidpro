from pathlib import Path
from typing import List

from painvidpro.data_storage.hdf5_video_archive import DynamicVideoArchive


def clean_ref_frame_tags(video_dir_list: List[str], tags_dir="reference_frame_tags", frame_data_file="frame_data.h5"):
    """
    Removes the 'reference_frame_tags' entry from the metadata and deletes the corresponding tags directory for each video directory in the provided list.

    Args:
        video_dir_list (List[str]): List of paths to video directories.
        tags_dir (str, optional): Name of the directory containing reference frame tags to be removed. Defaults to "reference_frame_tags".
        tags_dir (str): Name of the DynamicVideoArchive.

    Side Effects:
        - Modifies the metadata by removing the 'reference_frame_tags' key if present.
        - Deletes the tags directory within each video directory.
    """
    for video_dir in video_dir_list:
        frame_data_path = Path(video_dir) / frame_data_file
        with DynamicVideoArchive(filename=str(frame_data_path)) as archive:
            metadata = archive.get_global_metadata()
            if tags_dir in metadata:
                metadata.pop(tags_dir)
                archive.set_global_metadata(tags_dir, [])

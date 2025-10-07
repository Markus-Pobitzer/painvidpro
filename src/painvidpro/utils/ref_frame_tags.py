from pathlib import Path
from typing import List

from painvidpro.utils.metadata import load_metadata, save_metadata


def clean_ref_frame_tags(video_dir_list: List[str], tags_dir="reference_frame_tags"):
    """
    Removes the 'reference_frame_tags' entry from the metadata and deletes the corresponding tags directory for each video directory in the provided list.

    Args:
        video_dir_list (List[str]): List of paths to video directories.
        tags_dir (str, optional): Name of the directory containing reference frame tags to be removed. Defaults to "reference_frame_tags".

    Side Effects:
        - Modifies the metadata by removing the 'reference_frame_tags' key if present.
        - Deletes the tags directory within each video directory.
    """
    for video_dir in video_dir_list:
        vd = Path(video_dir)
        succ, metadata = load_metadata(vd)
        if "reference_frame_tags" in metadata:
            metadata.pop("reference_frame_tags")
            save_metadata(video_dir=vd, metadata=metadata)

import shutil
from pathlib import Path
from typing import List

from painvidpro.utils.metadata import load_metadata, save_metadata


def clean_ref_frame_variations(video_dir_list: List[str], variations_dir="reference_frame_variations"):
    """
    Removes the 'reference_frame_variations' entry from the metadata and deletes the corresponding variations directory for each video directory in the provided list.

    Args:
        video_dir_list (List[str]): List of paths to video directories.
        variations_dir (str, optional): Name of the directory containing reference frame variations to be removed. Defaults to "reference_frame_variations".

    Side Effects:
        - Modifies the metadata by removing the 'reference_frame_variations' key if present.
        - Deletes the variations directory within each video directory.
    """
    for video_dir in video_dir_list:
        vd = Path(video_dir)
        succ, metadata = load_metadata(vd)
        if "reference_frame_variations" in metadata:
            metadata.pop("reference_frame_variations")
            save_metadata(video_dir=vd, metadata=metadata)
        shutil.rmtree(vd / variations_dir, ignore_errors=True)

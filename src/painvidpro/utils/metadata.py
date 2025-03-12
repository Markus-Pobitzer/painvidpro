"""Utility for metadata loading and saving."""

import json
from pathlib import Path
from typing import Dict, Tuple, Union


def load_metadata(video_dir: Union[Path, str], metadata_name: str = "metadata.json") -> Tuple[bool, Dict]:
    """Loads the metadata from disk.

    Args:
        video_dir: Path to the folder contianing the metadata.
        metadata_name: Name of the metadata file.

    Returns:
        A bool if it was successfull and the metadata. If the success bool is
        false, an empty dict will be returned.
    """
    metadata_path = Path(video_dir) / metadata_name
    if metadata_path.exists():
        try:
            with open(metadata_path, "r") as f:
                return True, json.load(f)
        except Exception:
            return False, {}
    return False, {}


def save_metadata(video_dir: Path, metadata: Dict, metadata_name: str = "metadata.json") -> None:
    """Saves the metadata dict to disk.

    Args:
        video_dir: The path to the folder where the dict should be saved.
        metadata: The metadata dict.
        metadata_name: The name of the metadata.
    """
    metadata_path = video_dir / metadata_name
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

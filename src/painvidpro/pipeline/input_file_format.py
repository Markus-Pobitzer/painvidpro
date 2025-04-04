"""Dataclass for the input file for the pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class VideoProcessor:
    name: str
    config: Dict[str, Any]


@dataclass
class VideoItem:
    source: str
    url: str
    video_processor: List[VideoProcessor]
    channel_ids: List[str]
    art_style: List[str] = field(default_factory=list)
    art_genre: List[str] = field(default_factory=list)
    art_media: List[str] = field(default_factory=list)
    is_playlist: bool = False
    only_channel_vid: bool = False  # Only load videos from a playlist which channel is in channel_ids

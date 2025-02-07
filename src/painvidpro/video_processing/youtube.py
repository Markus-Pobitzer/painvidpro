"""File to download and process videos from YouTube."""

from typing import Any, Dict, List

import yt_dlp


def _extract_meta(result: Dict[str, Any], key_list: List[str], default: Any = "") -> Dict[str, Any]:
    """Extracts the keys from the results."""
    ret: Dict[str, Any] = {}
    for key in key_list:
        ret[key] = result.get(key, default)
    return ret


def download_video(url: str, output_path: str):
    """Downloads a YouTube video.

    Args:
        url: The YouTube URL.
        output_path: The output path on disk.
    """
    ydl_opts = {"format": "bestvideo[height<=360]", "outtmpl": output_path}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def get_ids_from_playlist(playlist_url: str) -> List[Dict[str, Any]]:
    """Get all Video Ids that are contained in a YouTube playlist.

    Args:
        playlist_url: The playlist URL.

    Returns:
        A list of YouTube metadata as a dict.
    """
    ydl_opts = {
        "outtmpl": "%(id)s%(title)s%(license)s%(channel_id)s%(channel)s",
        "quiet": True,
        "ignoreerrors": True,
    }
    ret: List[Dict[str, Any]] = []
    key_list = ["id", "title", "license", "channel", "channel_id"]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(playlist_url, download=False)

        if "entries" in result:
            for item in result["entries"]:
                ret.append(_extract_meta(item, key_list))
    return ret


def get_metadata_from_vido(video_id: str) -> Dict[str, Any]:
    """Extracts some metadata from the video.

    Args:
        video_id: The video ID or URL.

    Returns:
        A dict containing the metadata.
    """
    ydl_opts = {
        "outtmpl": "%(id)s%(title)s%(license)s%(channel_id)s%(channel)s",
        "quiet": True,
        "ignoreerrors": True,
    }
    key_list = ["id", "title", "license", "channel", "channel_id"]

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        res = ydl.extract_info(video_id, download=False)
        return _extract_meta(res, key_list)
